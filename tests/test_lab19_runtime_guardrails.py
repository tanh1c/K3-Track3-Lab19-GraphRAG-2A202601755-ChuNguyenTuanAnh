import json

import numpy as np
import pandas as pd

import lab19_runtime as runtime
from lab19_runtime import (
    RunConfig,
    apply_observed_suffix_aliases,
    ensure_entity_audit_schema,
    ensure_extraction_error_schema,
    valid_extraction_payload,
    valid_items_payload,
)


def _require_callable(name):
    fn = getattr(runtime, name, None)
    assert callable(fn), f"{name} is not implemented"
    return fn


class SequenceLLM:
    fast_model = "fake-model"
    generation_model = "fake-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected extra LLM call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, {"total_tokens": 1}


class FakeGraph:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, max_hops=2, edge_limit=50):
        self.calls.append(max_hops)
        return {
            "context": f"graph-hop-{max_hops}",
            "edges": pd.DataFrame([{"edge": max_hops}]),
            "diagnostics": {
                "matched_seeds": [{"id": "n1"}],
                "collected_edges": max_hops,
                "supernode_events": [],
            },
        }


class FakeFlat:
    def __init__(self):
        self.calls = 0

    def retrieve(self, query, k=8):
        self.calls += 1
        return "vector-fallback-context", pd.DataFrame([{"chunk_id": "v1"}])


class FakeEmbedder:
    def encode(self, texts, batch_size=128):
        rows = []
        for text in texts:
            text = str(text).lower()
            rows.append([1.0, 0.0] if "acquisition" in text else [0.0, 1.0])
        return np.asarray(rows, dtype="float32")


class FakeCommunityStore:
    def __init__(self):
        self.written = []

    def all_edges(self, limit=20_000):
        return pd.DataFrame(
            [
                {"source": "a", "source_name": "Alpha", "target": "b", "target_name": "Beta", "relation": "PARTNERED_WITH"},
                {"source": "c", "source_name": "Gamma", "target": "d", "target_name": "Delta", "relation": "ACQUIRED"},
            ]
        )

    def write_community_ids(self, rows):
        self.written.extend(rows)


def test_actual_runtime_config_uses_openai_without_fixed_groq_sleep():
    smoke = RunConfig.for_mode("smoke")
    full = RunConfig.for_mode("full")

    assert smoke.extraction_max_chunks == 24
    assert smoke.groq_timeout_s >= 60.0
    assert smoke.groq_min_interval_s <= 1.0
    assert smoke.coref_batch_size == 4
    assert smoke.extraction_batch_size == 4

    assert full.golden_limit == 50
    assert full.extraction_max_chunks == 400
    assert full.groq_timeout_s >= 60.0
    assert full.groq_min_interval_s <= 1.0


def test_empty_entity_resolution_audit_has_stable_csv_schema():
    audit = ensure_entity_audit_schema(pd.DataFrame())
    assert list(audit.columns) == ["type", "left", "right", "similarity", "decision"]
    assert audit.empty


def test_empty_extraction_errors_have_stable_csv_schema():
    errors = ensure_extraction_error_schema(pd.DataFrame())
    assert list(errors.columns) == ["start", "chunk_ids", "provider", "error"]
    assert errors.empty


def test_items_payload_guard_rejects_semantically_malformed_json():
    assert valid_items_payload({"items": [{"chunk_id": "c1", "relations": []}]}) is True
    assert valid_items_payload({"items": ["not-an-object"]}) is False
    assert valid_items_payload({"items": "not-a-list"}) is False


def test_extraction_payload_requires_every_expected_chunk_and_relation_dicts():
    expected = {"c1", "c2"}
    assert valid_extraction_payload(
        {"items": [{"chunk_id": "c1", "relations": []}, {"chunk_id": "c2", "relations": [{"source": "A"}]}]},
        expected,
    ) is True
    assert valid_extraction_payload({"items": [{"chunk_id": "c1", "relations": []}]}, expected) is False
    assert valid_extraction_payload(
        {"items": [{"chunk_id": "c1", "relations": []}, {"chunk_id": "c2", "relations": ["bad"]}]},
        expected,
    ) is False


def test_observed_company_suffix_variants_create_real_manual_merge_audit():
    raw = pd.DataFrame(
        [
            {
                "source_type": "Company",
                "source_raw": "Acme Inc.",
                "target_type": "Technology",
                "target_raw": "Widget AI",
            },
            {
                "source_type": "Company",
                "source_raw": "Acme Corporation",
                "target_type": "Technology",
                "target_raw": "Cloud Stack",
            },
        ]
    )
    mapping = {
        ("Company", "acme inc."): "Acme Inc.",
        ("Company", "acme corporation"): "Acme Corporation",
        ("Technology", "widget ai"): "Widget AI",
        ("Technology", "cloud stack"): "Cloud Stack",
    }
    out_mapping, audit = apply_observed_suffix_aliases(raw, mapping, pd.DataFrame())
    assert out_mapping[("Company", "acme inc.")] == out_mapping[("Company", "acme corporation")]
    assert "MERGE_MANUAL" in set(audit.decision)


def test_llm_self_correction_stops_at_hop2_when_context_is_sufficient():
    fn = _require_callable("self_correcting_context_llm")
    graph = FakeGraph()
    flat = FakeFlat()
    llm = SequenceLLM([{"sufficient": True, "missing": ""}])

    result = fn("question", graph, flat, llm)

    assert result["route"] == "hop2"
    assert result["stop_condition"] == "SUFFICIENT_HOP2"
    assert graph.calls == [2]
    assert flat.calls == 0
    assert len(llm.calls) == 1


def test_llm_self_correction_expands_to_hop3_then_vector_and_stops():
    fn = _require_callable("self_correcting_context_llm")
    graph = FakeGraph()
    flat = FakeFlat()
    llm = SequenceLLM(
        [
            {"sufficient": False, "missing": "partner evidence"},
            {"sufficient": False, "missing": "second document"},
            {"sufficient": True, "missing": ""},
        ]
    )

    result = fn("question", graph, flat, llm)

    assert result["route"] == "hop3+vector"
    assert result["stop_condition"] == "VECTOR_FALLBACK_STOP"
    assert graph.calls == [2, 3]
    assert flat.calls == 1
    assert result["hop2_sufficient"] is False
    assert result["final_sufficient"] is True
    assert "vector-fallback-context" in result["context"]
    assert len(llm.calls) == 3


def test_networkx_community_fallback_writes_ids_and_uses_llm_summaries(tmp_path):
    fn = _require_callable("build_llm_community_reports")
    store = FakeCommunityStore()
    nodes = pd.DataFrame(
        [
            {"id": "a", "name": "Alpha"},
            {"id": "b", "name": "Beta"},
            {"id": "c", "name": "Gamma"},
            {"id": "d", "name": "Delta"},
        ]
    )
    llm = SequenceLLM(
        [
            {
                "items": [
                    {"community_id": 0, "summary": "Alpha and Beta form a partnership cluster."},
                    {"community_id": 1, "summary": "Gamma acquired Delta in an acquisition cluster."},
                ]
            }
        ]
    )

    reports = fn(store, nodes, tmp_path, llm, batch_size=16)

    assert len(store.written) == 4
    assert set(reports["summary_method"]) == {"LLM"}
    assert reports["report"].str.len().min() > 20
    assert (tmp_path / "community_reports.csv").exists()


def test_query_router_dispatches_global_queries_to_community_reports():
    fn = _require_callable("route_bonus_query")
    graph = FakeGraph()
    flat = FakeFlat()
    llm = SequenceLLM([{"route": "global", "reason": "corpus-level themes"}])
    reports = pd.DataFrame(
        [
            {"community_id": 0, "size": 3, "hubs": "A | B", "dominant_relations": "ACQUIRED", "report": "acquisition ecosystem"},
            {"community_id": 1, "size": 2, "hubs": "C | D", "dominant_relations": "USES", "report": "AI technology ecosystem"},
        ]
    )

    result = fn(
        "Across the corpus, what are the major acquisition themes?",
        graph,
        flat,
        reports,
        FakeEmbedder(),
        llm,
    )

    assert result["route"] == "global"
    assert result["retrieval_level"] == "high-level-community"
    assert result["retrieved_items"] > 0
    assert graph.calls == []
    assert flat.calls == 0


def test_malformed_extraction_batch_is_repaired_per_item_before_recording_error(tmp_path):
    fn = _require_callable("extract_triples_with_repair")
    source = pd.DataFrame(
        [
            {
                "chunk_id": "c1",
                "source_row_id": 7,
                "published_date": "2023-01-01",
                "resolved_text": "Alpha partnered with Beta.",
            }
        ]
    )
    llm = SequenceLLM(
        [
            {"items": "malformed"},
            {"items": [{"chunk_id": "c1", "relations": []}]},
        ]
    )

    triples, errors = fn(source, llm, RunConfig.for_mode("smoke"), tmp_path)

    assert triples.empty
    assert errors.empty
    assert len(llm.calls) == 2
    checkpoint_rows = [json.loads(line) for line in (tmp_path / "checkpoints" / "extraction.jsonl").read_text().splitlines()]
    assert checkpoint_rows[-1]["chunk_id"] == "c1"
