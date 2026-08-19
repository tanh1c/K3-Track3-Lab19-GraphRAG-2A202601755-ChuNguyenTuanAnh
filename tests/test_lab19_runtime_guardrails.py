import pandas as pd

from lab19_runtime import (
    RunConfig,
    apply_observed_suffix_aliases,
    ensure_entity_audit_schema,
    ensure_extraction_error_schema,
    valid_extraction_payload,
    valid_items_payload,
)


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
