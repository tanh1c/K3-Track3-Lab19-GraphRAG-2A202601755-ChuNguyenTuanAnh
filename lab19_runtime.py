"""Runtime guardrails and strict bonus implementation for Lab 19.

The instructor implementation remains in :mod:`lab19_runtime_impl`. This wrapper
keeps that code inspectable while making OpenAI the primary LLM provider and
adds rubric-facing production guardrails without changing the core benchmark
contract.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

import lab19_runtime_impl as _impl
from lab19_core import merge_guard, normalize_entity, strip_corporate_suffix
from lab19_models import (
    DEFAULT_LLM_PROVIDER,
    DEFAULT_OPENAI_GENERATION_MODEL,
    DEFAULT_OPENAI_PIPELINE_MODEL,
)

os.environ.setdefault("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
os.environ.setdefault("OPENAI_PIPELINE_MODEL", DEFAULT_OPENAI_PIPELINE_MODEL)
os.environ.setdefault("OPENAI_GENERATION_MODEL", DEFAULT_OPENAI_GENERATION_MODEL)

ENTITY_AUDIT_COLUMNS = ["type", "left", "right", "similarity", "decision"]
EXTRACTION_ERROR_COLUMNS = ["start", "chunk_ids", "provider", "error"]
_ACTIVE_LLM: Any | None = None
_BONUS_COMPONENTS: dict[str, Any] = {}


def ensure_entity_audit_schema(audit: pd.DataFrame) -> pd.DataFrame:
    if audit is None or audit.empty:
        return pd.DataFrame(columns=ENTITY_AUDIT_COLUMNS)
    out = audit.copy()
    for column in ENTITY_AUDIT_COLUMNS:
        if column not in out.columns:
            out[column] = None
    ordered = ENTITY_AUDIT_COLUMNS + [c for c in out.columns if c not in ENTITY_AUDIT_COLUMNS]
    return out[ordered]


def ensure_extraction_error_schema(errors: pd.DataFrame) -> pd.DataFrame:
    if errors is None or errors.empty:
        return pd.DataFrame(columns=EXTRACTION_ERROR_COLUMNS)
    out = errors.copy()
    for column in EXTRACTION_ERROR_COLUMNS:
        if column not in out.columns:
            out[column] = None
    ordered = EXTRACTION_ERROR_COLUMNS + [c for c in out.columns if c not in EXTRACTION_ERROR_COLUMNS]
    return out[ordered]


def valid_items_payload(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if "items" not in obj:
        return True
    items = obj.get("items")
    return isinstance(items, list) and all(isinstance(item, dict) for item in items)


def valid_extraction_payload(obj: Any, expected_chunk_ids: set[str]) -> bool:
    if not valid_items_payload(obj):
        return False
    items = obj.get("items")
    if not isinstance(items, list):
        return False
    seen: set[str] = set()
    for item in items:
        cid = str(item.get("chunk_id", ""))
        relations = item.get("relations")
        if not cid or not isinstance(relations, list):
            return False
        if not all(isinstance(relation, dict) for relation in relations):
            return False
        seen.add(cid)
    return expected_chunk_ids.issubset(seen)


_original_for_mode = _impl.RunConfig.for_mode


def _safe_for_mode(cls, mode: str):
    base = _original_for_mode(mode)
    common = {
        # Legacy field names remain for notebook compatibility; values now govern
        # the primary OpenAI client instead of Groq.
        "groq_timeout_s": max(float(base.groq_timeout_s), 60.0),
        "groq_min_interval_s": 0.0,
        "groq_max_retries": 5,
        "coref_batch_size": 4,
        "extraction_batch_size": 4,
    }
    if str(base.mode).lower() == "full":
        return replace(base, extraction_max_chunks=400, **common)
    return replace(base, **common)


_impl.RunConfig.for_mode = classmethod(_safe_for_mode)


class OpenAIRuntime:
    """OpenAI-backed runtime with SDK retries and no artificial provider sleep."""

    def __init__(self, config: Any):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
        if provider != "openai":
            raise RuntimeError(f"Unsupported primary LLM_PROVIDER={provider!r}; expected 'openai'")
        self.fast_model = (
            os.getenv("OPENAI_PIPELINE_MODEL", DEFAULT_OPENAI_PIPELINE_MODEL).strip()
            or DEFAULT_OPENAI_PIPELINE_MODEL
        )
        self.generation_model = (
            os.getenv("OPENAI_GENERATION_MODEL", DEFAULT_OPENAI_GENERATION_MODEL).strip()
            or self.fast_model
        )
        self.config = config
        self.client = OpenAI(
            api_key=api_key,
            timeout=max(float(config.groq_timeout_s), 60.0),
            max_retries=max(int(config.groq_max_retries), 3),
        )
        global _ACTIVE_LLM
        _ACTIVE_LLM = self

    def _pace(self) -> None:
        return None

    @staticmethod
    def parse_json(text: str) -> dict[str, Any]:
        return _impl.GroqRuntimeOriginal.parse_json(text) if hasattr(_impl, "GroqRuntimeOriginal") else json.loads(text)

    def chat(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 1200,
        json_mode: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "model": model or self.fast_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": int(max_tokens),
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        usage_obj = getattr(response, "usage", None)
        usage = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
            "completion_tokens": getattr(usage_obj, "completion_tokens", None),
            "total_tokens": getattr(usage_obj, "total_tokens", None),
        }
        return (response.choices[0].message.content or "").strip(), usage

    def chat_json(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        text, usage = self.chat(json_mode=True, **kwargs)
        clean = str(text).strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1]
            clean = clean.rsplit("```", 1)[0]
        obj = json.loads(clean)
        if not valid_items_payload(obj):
            raise ValueError("Malformed items envelope")
        return obj, usage


# Preserve the old parser for tests/backward compatibility before replacing the
# instructor's provider class globally.
_impl.GroqRuntimeOriginal = _impl.GroqRuntime
_impl.GroqRuntime = OpenAIRuntime


def apply_observed_suffix_aliases(
    raw_triples_df: pd.DataFrame,
    mapping: dict[tuple[str, str], str],
    audit: pd.DataFrame,
) -> tuple[dict[tuple[str, str], str], pd.DataFrame]:
    out_mapping = dict(mapping)
    rows = ensure_entity_audit_schema(audit).to_dict("records")
    mentions: list[tuple[str, str]] = []
    for r in raw_triples_df.itertuples(index=False):
        mentions.extend([(str(r.source_type), str(r.source_raw)), (str(r.target_type), str(r.target_raw))])
    counts = Counter((typ, normalize_entity(name)) for typ, name in mentions)
    display: dict[tuple[str, str], str] = {}
    for typ, name in mentions:
        display.setdefault((typ, normalize_entity(name)), name)

    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in counts:
        typ, _ = key
        if typ != "Company":
            continue
        base = strip_corporate_suffix(display[key])
        if base:
            groups[base].append(key)

    existing = {
        (str(row.get("decision")), normalize_entity(row.get("left")), normalize_entity(row.get("right")))
        for row in rows
    }
    for keys in groups.values():
        unique = list(dict.fromkeys(keys))
        if len(unique) < 2:
            continue
        canonical_key = sorted(unique, key=lambda key: (-counts[key], len(display[key]), display[key].lower()))[0]
        canonical = display[canonical_key]
        for key in unique:
            out_mapping[key] = canonical
            if key == canonical_key:
                continue
            marker = ("MERGE_MANUAL", normalize_entity(display[key]), normalize_entity(canonical))
            if marker in existing:
                continue
            rows.append(
                {
                    "type": "Company",
                    "left": display[key],
                    "right": canonical,
                    "similarity": 1.0,
                    "decision": "MERGE_MANUAL",
                }
            )
            existing.add(marker)
    return out_mapping, ensure_entity_audit_schema(pd.DataFrame(rows))


def _append_guard_rejections(
    raw_triples_df: pd.DataFrame,
    embedder: Any,
    audit: pd.DataFrame,
    *,
    audit_threshold: float = 0.75,
    merge_threshold: float = 0.90,
    top_k: int = 5,
    max_rows: int = 100,
) -> pd.DataFrame:
    try:
        import faiss
    except Exception:
        return ensure_entity_audit_schema(audit)

    rows = ensure_entity_audit_schema(audit).to_dict("records")
    existing_pairs = {
        (
            str(row.get("decision")),
            str(row.get("type")),
            tuple(sorted((normalize_entity(row.get("left")), normalize_entity(row.get("right"))))),
        )
        for row in rows
    }
    mentions_by_type: dict[str, dict[str, str]] = defaultdict(dict)
    for r in raw_triples_df.itertuples(index=False):
        for typ, name in [(str(r.source_type), str(r.source_raw)), (str(r.target_type), str(r.target_raw))]:
            norm = normalize_entity(name)
            if norm:
                mentions_by_type[typ].setdefault(norm, name)

    for typ in sorted(mentions_by_type):
        names = list(mentions_by_type[typ].values())
        if len(names) < 2:
            continue
        vectors = embedder.encode(names)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        sims, nbrs = index.search(vectors, min(top_k, len(names)))
        for i in range(len(names)):
            for score, j in zip(sims[i], nbrs[i]):
                j = int(j)
                score = float(score)
                if j < 0 or i >= j or score < audit_threshold or score >= merge_threshold:
                    continue
                if merge_guard(names[i], names[j], typ):
                    continue
                pair = tuple(sorted((normalize_entity(names[i]), normalize_entity(names[j]))))
                marker = ("REJECT_GUARD", typ, pair)
                if marker in existing_pairs:
                    continue
                rows.append(
                    {
                        "type": typ,
                        "left": names[i],
                        "right": names[j],
                        "similarity": score,
                        "decision": "REJECT_GUARD",
                    }
                )
                existing_pairs.add(marker)
                if len(rows) >= max_rows:
                    return ensure_entity_audit_schema(pd.DataFrame(rows))
    return ensure_entity_audit_schema(pd.DataFrame(rows))


_original_build_resolution_map = _impl.build_resolution_map


def _guarded_build_resolution_map(*args: Any, **kwargs: Any):
    mapping, audit = _original_build_resolution_map(*args, **kwargs)
    raw_triples_df = args[0] if args else kwargs["raw_triples_df"]
    embedder = args[1] if len(args) > 1 else kwargs["embedder"]
    mapping, audit = apply_observed_suffix_aliases(raw_triples_df, mapping, audit)
    audit = _append_guard_rejections(raw_triples_df, embedder, audit)
    return mapping, ensure_entity_audit_schema(audit)


_impl.build_resolution_map = _guarded_build_resolution_map


def _extraction_prompt(payload: list[dict[str, Any]]) -> str:
    return (
        "Return {\"items\":[{\"chunk_id\":\"...\",\"relations\":[{"
        "\"source\":\"...\",\"source_type\":\"Company|Person|Technology\","
        "\"relation\":\"ALLOWED_RELATION\",\"target\":\"...\","
        "\"target_type\":\"Company|Person|Technology\",\"evidence\":\"...\","
        "\"confidence\":0.0}]}]}. Include one items entry for EVERY input chunk, even when relations is empty. INPUT:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def extract_triples_with_repair(
    source_df: pd.DataFrame,
    llm: Any,
    config: Any,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract triples with resumable checkpoints and per-item JSON repair.

    A malformed/incomplete batch is never checkpointed as complete. The batch is
    retried as single-item requests so one bad envelope cannot silently discard
    four extraction chunks.
    """
    checkpoint = _impl.JsonlCheckpoint(output_dir / "checkpoints" / "extraction.jsonl", "chunk_id")
    completed = checkpoint.completed_keys()
    errors: list[dict[str, Any]] = []
    system = (
        "Extract a HIGH-PRECISION knowledge graph from technology-news text. "
        f"Allowed node types: {sorted(_impl.ALLOWED_NODE_TYPES)}. "
        f"Allowed relations: {sorted(_impl.ALLOWED_RELATIONS)}. "
        "Use only explicitly supported facts. Prefer precision over recall. "
        "Every relation MUST include a short verbatim evidence span and confidence 0..1. Return strict JSON."
    )
    pending = source_df[~source_df["chunk_id"].isin(completed)].copy()
    for start in range(0, len(pending), config.extraction_batch_size):
        batch = pending.iloc[start : start + config.extraction_batch_size]
        payload = [
            {"chunk_id": r.chunk_id, "published_date": r.published_date, "text": r.resolved_text}
            for r in batch.itertuples(index=False)
        ]
        expected_ids = {str(item["chunk_id"]) for item in payload}
        by_id: dict[str, dict[str, Any]] = {}
        batch_error: Exception | None = None
        try:
            obj, _ = llm.chat_json(
                system=system,
                user=_extraction_prompt(payload),
                model=llm.fast_model,
                max_tokens=1800,
            )
            if not valid_extraction_payload(obj, expected_ids):
                raise ValueError("Malformed or incomplete extraction items envelope")
            by_id = {str(item.get("chunk_id")): item for item in obj.get("items", [])}
        except Exception as exc:
            batch_error = exc
            print(f"[extract] repairing malformed batch item-by-item: {type(exc).__name__}: {exc}")
            for one in payload:
                cid = str(one["chunk_id"])
                try:
                    obj, _ = llm.chat_json(
                        system=system,
                        user=_extraction_prompt([one]),
                        model=llm.fast_model,
                        max_tokens=1000,
                    )
                    if not valid_extraction_payload(obj, {cid}):
                        raise ValueError("Malformed single-item extraction envelope")
                    by_id[cid] = next(item for item in obj["items"] if str(item.get("chunk_id")) == cid)
                except Exception as repair_exc:
                    print(f"[extract] single-item repair failed for {cid}: {type(repair_exc).__name__}: {repair_exc}")

        for row in batch.itertuples(index=False):
            cid = str(row.chunk_id)
            if cid in by_id:
                checkpoint.upsert(by_id[cid])

        missing = sorted(expected_ids.difference(by_id))
        if missing:
            errors.append(
                {
                    "start": int(start),
                    "chunk_ids": json.dumps(missing),
                    "provider": "openai",
                    "error": (
                        f"{type(batch_error).__name__}: {batch_error}; unrepaired={len(missing)}"
                        if batch_error is not None
                        else f"Unrepaired extraction chunks={len(missing)}"
                    ),
                }
            )

    meta = source_df.set_index("chunk_id").to_dict("index")
    triples: list[dict[str, Any]] = []
    for item in checkpoint.rows():
        cid = str(item.get("chunk_id", ""))
        if cid not in meta:
            continue
        info = meta[cid]
        published_date = _impl.normalize_text(info.get("published_date"))
        if not published_date:
            continue
        relations = item.get("relations", []) or []
        if not isinstance(relations, list):
            continue
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            source = _impl.normalize_text(relation.get("source"))
            target = _impl.normalize_text(relation.get("target"))
            st = relation.get("source_type")
            tt = relation.get("target_type")
            rel = relation.get("relation")
            evidence = _impl.normalize_text(relation.get("evidence"))
            try:
                confidence = float(relation.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = -1.0
            if not source or not target or not evidence:
                continue
            if st not in _impl.ALLOWED_NODE_TYPES or tt not in _impl.ALLOWED_NODE_TYPES:
                continue
            if rel not in _impl.ALLOWED_RELATIONS or not (0 <= confidence <= 1):
                continue
            triples.append(
                {
                    "source_raw": source,
                    "source_type": st,
                    "relation": rel,
                    "target_raw": target,
                    "target_type": tt,
                    "source_chunk_id": cid,
                    "source_row_id": int(info["source_row_id"]),
                    "published_date": published_date,
                    "evidence": evidence,
                    "confidence": confidence,
                }
            )

    columns = [
        "source_raw", "source_type", "relation", "target_raw", "target_type",
        "source_chunk_id", "source_row_id", "published_date", "evidence", "confidence",
    ]
    triples_df = pd.DataFrame(triples, columns=columns)
    triples_df = _impl.validate_extracted_triples(triples_df)
    error_df = ensure_extraction_error_schema(pd.DataFrame(errors))
    triples_df.to_csv(output_dir / "raw_triples.csv", index=False)
    error_df.to_csv(output_dir / "extraction_errors.csv", index=False)
    print(f"[extract] valid triples={len(triples_df):,}; errors={len(error_df):,}; provider=openai")
    return triples_df, error_df


_impl.extract_triples = extract_triples_with_repair


# ---------- Strict Bonus C: LLM sufficiency + bounded self-correction ----------

SUFFICIENCY_SYSTEM = (
    "Decide whether the supplied retrieval context is sufficient to answer the question faithfully. "
    "Do not answer the question. Return strict JSON only with keys sufficient and missing."
)


def llm_context_sufficient(llm: Any, question: str, context: str) -> tuple[bool, str]:
    try:
        obj, _ = llm.chat_json(
            system=SUFFICIENCY_SYSTEM,
            user=(
                f"QUESTION: {question}\nCONTEXT:\n{str(context)[:16000]}\n"
                'Return {"sufficient":true,"missing":"..."}'
            ),
            model=llm.fast_model,
            max_tokens=300,
        )
        raw = obj.get("sufficient", False)
        if isinstance(raw, bool):
            sufficient = raw
        else:
            sufficient = str(raw).strip().lower() in {"true", "1", "yes"}
        return sufficient, _impl.normalize_text(obj.get("missing"))
    except Exception as exc:
        return False, f"sufficiency_check_error:{type(exc).__name__}"


def self_correcting_context_llm(
    question: str,
    graph: Any,
    flat: Any,
    llm: Any,
) -> dict[str, Any]:
    """Bonus retrieval with a hard stop: hop2 -> hop3 -> vector fallback -> stop."""
    hop2 = graph.retrieve(question, max_hops=2, edge_limit=50)
    ok2, missing2 = llm_context_sufficient(llm, question, hop2.get("context", ""))
    hop2_chars = len(str(hop2.get("context", "")))
    checks = [{"stage": "hop2", "sufficient": ok2, "missing": missing2}]
    if ok2:
        return {
            **hop2,
            "route": "hop2",
            "stop_condition": "SUFFICIENT_HOP2",
            "missing": "",
            "hop2_sufficient": True,
            "final_sufficient": True,
            "hop2_context_chars": hop2_chars,
            "final_context_chars": hop2_chars,
            "sufficiency_checks": checks,
        }

    hop3 = graph.retrieve(question, max_hops=3, edge_limit=50)
    ok3, missing3 = llm_context_sufficient(llm, question, hop3.get("context", ""))
    checks.append({"stage": "hop3", "sufficient": ok3, "missing": missing3})
    hop3_chars = len(str(hop3.get("context", "")))
    if ok3:
        return {
            **hop3,
            "route": "hop3",
            "stop_condition": "SUFFICIENT_HOP3",
            "missing": missing2,
            "hop2_sufficient": False,
            "final_sufficient": True,
            "hop2_context_chars": hop2_chars,
            "final_context_chars": hop3_chars,
            "sufficiency_checks": checks,
        }

    vector_context, vector_docs = flat.retrieve(question, k=8)
    final_context = f"=== GRAPH ===\n{hop3.get('context', '')}\n\n=== VECTOR ===\n{vector_context}"
    final_ok, final_missing = llm_context_sufficient(llm, question, final_context)
    checks.append({"stage": "hop3+vector-final-check", "sufficient": final_ok, "missing": final_missing})
    return {
        **hop3,
        "route": "hop3+vector",
        "context": final_context,
        "vector_docs": vector_docs,
        "stop_condition": "VECTOR_FALLBACK_STOP",
        "missing": missing3,
        "hop2_sufficient": False,
        "final_sufficient": bool(final_ok),
        "hop2_context_chars": hop2_chars,
        "final_context_chars": len(final_context),
        "sufficiency_checks": checks,
    }


# ---------- Strict Bonus A/B: NetworkX communities, LLM reports, query router ----------


def _python_connected_components(node_ids: list[str], edge_df: pd.DataFrame) -> list[set[str]]:
    adjacency: dict[str, set[str]] = {str(node): set() for node in node_ids}
    for row in edge_df.itertuples(index=False):
        source, target = str(row.source), str(row.target)
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    seen: set[str] = set()
    groups: list[set[str]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        members: set[str] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            members.add(node)
            stack.extend(sorted(adjacency.get(node, ()), reverse=True))
        groups.append(members)
    return groups


def build_llm_community_reports(
    store: Any,
    nodes_df: pd.DataFrame,
    output_dir: Path,
    llm: Any,
    *,
    batch_size: int = 16,
) -> pd.DataFrame:
    """NetworkX fallback: detect communities, write ids, then LLM-summarize each community."""
    edge_df = store.all_edges(limit=20_000)
    node_ids = [str(value) for value in nodes_df.get("id", pd.Series(dtype=str)).tolist()]
    algorithm = "networkx.greedy_modularity_communities"
    try:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(node_ids)
        if not edge_df.empty:
            graph.add_edges_from(edge_df[["source", "target"]].itertuples(index=False, name=None))
        if graph.number_of_edges() == 0:
            communities = [{str(node)} for node in graph.nodes]
        else:
            communities = [set(map(str, group)) for group in nx.algorithms.community.greedy_modularity_communities(graph)]
        degree = dict(graph.degree())
    except ImportError:
        algorithm = "python.connected_components_test_fallback"
        communities = _python_connected_components(node_ids, edge_df)
        degree = Counter()
        if not edge_df.empty:
            for row in edge_df.itertuples(index=False):
                degree[str(row.source)] += 1
                degree[str(row.target)] += 1

    node_lookup = {
        str(row.id): str(row.name)
        for row in nodes_df[["id", "name"]].itertuples(index=False)
    }
    membership: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    for cid, members in enumerate(communities):
        members = set(map(str, members))
        membership.extend({"id": node_id, "community_id": int(cid)} for node_id in sorted(members))
        hubs_ids = sorted(members, key=lambda node: (-int(degree.get(node, 0)), node))[:8]
        hubs = [node_lookup.get(node, node) for node in hubs_ids]
        if edge_df.empty:
            internal = edge_df.copy()
        else:
            internal = edge_df[edge_df["source"].astype(str).isin(members) & edge_df["target"].astype(str).isin(members)].copy()
        relation_counts = Counter(internal["relation"].astype(str).tolist()) if not internal.empty else Counter()
        dominant = [name for name, _ in relation_counts.most_common(5)]
        edge_examples = []
        for edge in internal.head(12).itertuples(index=False):
            edge_examples.append(
                f"{getattr(edge, 'source_name', edge.source)} -{edge.relation}-> {getattr(edge, 'target_name', edge.target)}"
            )
        fallback = (
            f"Community {cid} contains {len(members)} entities. "
            f"Main hubs: {', '.join(hubs)}. Dominant relations: {', '.join(dominant) or 'none'}."
        )
        report_rows.append(
            {
                "community_id": int(cid),
                "size": len(members),
                "hubs": " | ".join(hubs),
                "dominant_relations": " | ".join(dominant),
                "edge_examples": " | ".join(edge_examples),
                "report": fallback,
                "summary_method": "FALLBACK_DETERMINISTIC",
                "community_algorithm": algorithm,
            }
        )

    store.write_community_ids(membership)
    system = (
        "Summarize knowledge-graph communities using ONLY the supplied entities and relation examples. "
        "Describe the high-level topic/theme and important relationships. Do not invent facts. Return strict JSON."
    )
    for start in range(0, len(report_rows), max(1, int(batch_size))):
        batch = report_rows[start : start + max(1, int(batch_size))]
        payload = [
            {
                "community_id": row["community_id"],
                "size": row["size"],
                "hubs": row["hubs"],
                "dominant_relations": row["dominant_relations"],
                "edge_examples": row["edge_examples"],
            }
            for row in batch
        ]
        try:
            obj, _ = llm.chat_json(
                system=system,
                user=(
                    'Return {"items":[{"community_id":0,"summary":"..."}]}, exactly one item per input community. INPUT:\n'
                    + json.dumps(payload, ensure_ascii=False)
                ),
                model=llm.fast_model,
                max_tokens=2400,
            )
            summaries = {
                int(item.get("community_id")): _impl.normalize_text(item.get("summary"))
                for item in obj.get("items", [])
                if isinstance(item, dict) and str(item.get("community_id", "")).lstrip("-").isdigit()
            }
            for row in batch:
                summary = summaries.get(int(row["community_id"]), "")
                if summary:
                    row["report"] = summary
                    row["summary_method"] = "LLM"
        except Exception as exc:
            print(f"[bonus-community] LLM batch fallback: {type(exc).__name__}: {exc}")

    reports_df = pd.DataFrame(report_rows)
    if not reports_df.empty:
        reports_df = reports_df.sort_values(["size", "community_id"], ascending=[False, True]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_df.to_csv(output_dir / "community_reports.csv", index=False)
    return reports_df


def _build_community_reports_active(store: Any, nodes_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if _ACTIVE_LLM is None:
        raise RuntimeError("Community LLM summarization requires an active pipeline LLM")
    return build_llm_community_reports(store, nodes_df, output_dir, _ACTIVE_LLM)


_impl.build_community_reports = _build_community_reports_active


ROUTER_SYSTEM = (
    "Route a GraphRAG query to exactly one retrieval level. Use 'local' for questions about a specific named "
    "entity, event, relation or short path. Use 'global' for corpus-wide themes, patterns, communities, trends or "
    "comparisons spanning many unrelated entities. Return strict JSON with route and reason."
)


def classify_query_level(question: str, llm: Any) -> tuple[str, str]:
    try:
        obj, _ = llm.chat_json(
            system=ROUTER_SYSTEM,
            user=f'QUESTION: {question}\nReturn {{"route":"local|global","reason":"..."}}',
            model=llm.fast_model,
            max_tokens=220,
        )
        route = str(obj.get("route", "")).strip().lower()
        reason = _impl.normalize_text(obj.get("reason"))
        if route in {"local", "global"}:
            return route, reason
    except Exception as exc:
        reason = f"router_error:{type(exc).__name__}"
    broad_markers = ("across the corpus", "overall", "major themes", "communities", "patterns", "trends")
    route = "global" if any(marker in question.lower() for marker in broad_markers) else "local"
    return route, reason if 'reason' in locals() else "deterministic fallback"


def route_bonus_query(
    question: str,
    graph: Any,
    flat: Any,
    reports_df: pd.DataFrame,
    embedder: Any,
    llm: Any,
) -> dict[str, Any]:
    route, reason = classify_query_level(question, llm)
    if route == "global" and not reports_df.empty:
        vectors = embedder.encode(reports_df["report"].fillna("").astype(str).tolist())
        query_vector = embedder.encode([question], batch_size=1)[0]
        scores = vectors @ query_vector
        order = list(scores.argsort()[::-1][: min(5, len(scores))])
        hits = reports_df.iloc[order].copy()
        hits["similarity"] = [float(scores[idx]) for idx in order]
        context = "\n\n".join(
            f"[community_id={row.community_id} | size={row.size}] {row.report}"
            for row in hits.itertuples(index=False)
        )
        return {
            "route": "global",
            "reason": reason,
            "retrieval_level": "high-level-community",
            "context": context,
            "retrieved_items": len(hits),
        }

    local = graph.retrieve(question, max_hops=2, edge_limit=50)
    context = local.get("context", "")
    retrieved = int(local.get("diagnostics", {}).get("collected_edges", 0))
    if not context:
        vector_context, vector_docs = flat.retrieve(question, k=8)
        context = vector_context
        retrieved = len(vector_docs)
    return {
        "route": "local",
        "reason": reason,
        "retrieval_level": "low-level-entity",
        "context": context,
        "retrieved_items": int(retrieved),
    }


# ---------- Evidence hooks: spot-checks, bonus ablation, report packaging ----------

_original_resolve_coreferences = _impl.resolve_coreferences


def _resolve_coreferences_with_spotcheck(source_df: pd.DataFrame, llm: Any, config: Any, output_dir: Path) -> pd.DataFrame:
    resolved = _original_resolve_coreferences(source_df, llm, config, output_dir)
    candidates = resolved[resolved.get("coref_route", pd.Series(index=resolved.index, dtype=str)).ne("NO_TRIGGER")].head(10).copy()
    if candidates.empty:
        candidates = resolved.head(10).copy()
    columns = [c for c in ["chunk_id", "source_row_id", "text", "resolved_text", "coref_route", "unresolved_mentions"] if c in candidates.columns]
    spot = candidates[columns].copy()
    if "unresolved_mentions" in spot.columns:
        spot["unresolved_mentions"] = spot["unresolved_mentions"].map(
            lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
        )
    spot.to_csv(output_dir / "coref_spotcheck.csv", index=False)
    return resolved


_impl.resolve_coreferences = _resolve_coreferences_with_spotcheck

_original_evaluate = _impl.evaluate


def _evaluate_and_capture(golden, flat, graph, llm, judge, config, output_dir):
    result = _original_evaluate(golden, flat, graph, llm, judge, config, output_dir)
    _BONUS_COMPONENTS.clear()
    _BONUS_COMPONENTS.update({"flat": flat, "graph": graph, "llm": llm, "config": config})
    return result


_impl.evaluate = _evaluate_and_capture

_original_write_bonus_metrics = _impl.write_bonus_metrics


def _write_bonus_metrics_strict(near_audit, community_reports, eval_df, output_dir):
    base = _original_write_bonus_metrics(near_audit, community_reports, eval_df, output_dir)
    flat = _BONUS_COMPONENTS.get("flat")
    graph = _BONUS_COMPONENTS.get("graph")
    llm = _BONUS_COMPONENTS.get("llm")
    if flat is None or graph is None or llm is None:
        return base

    golden = pd.read_csv(_impl.GOLDEN_PATH)
    per_group = 4 if len(eval_df) >= 50 else 1
    sample_parts = [golden[golden["group"].eq(group)].head(per_group) for group in ["factoid", "multi-hop", "cross-doc"]]
    sample = pd.concat(sample_parts, ignore_index=True)
    correction_rows: list[dict[str, Any]] = []
    for row in sample.itertuples(index=False):
        result = self_correcting_context_llm(str(row.question), graph, flat, llm)
        correction_rows.append(
            {
                "id": row.id,
                "group": row.group,
                "question": row.question,
                "route": result.get("route"),
                "stop_condition": result.get("stop_condition"),
                "hop2_sufficient": bool(result.get("hop2_sufficient")),
                "final_sufficient": bool(result.get("final_sufficient")),
                "hop2_context_chars": int(result.get("hop2_context_chars", 0)),
                "final_context_chars": int(result.get("final_context_chars", 0)),
                "missing": result.get("missing", ""),
            }
        )
    correction_df = pd.DataFrame(correction_rows)
    correction_df.to_csv(output_dir / "self_correction_bonus_eval.csv", index=False)

    router_cases: list[tuple[str, str]] = []
    for group in ["factoid", "multi-hop"]:
        hit = golden[golden["group"].eq(group)].head(1)
        if not hit.empty:
            router_cases.append((f"golden-{group}", str(hit.iloc[0].question)))
    router_cases.extend(
        [
            ("broad-corpus", "Across the corpus, what are the major partnership and acquisition themes?"),
            ("broad-communities", "Which high-level knowledge-graph communities dominate the selected corpus, and how do their themes differ?"),
        ]
    )
    router_rows = []
    for source, question in router_cases:
        routed = route_bonus_query(question, graph, flat, community_reports, flat.embedder, llm)
        router_rows.append(
            {
                "query_source": source,
                "question": question,
                "route": routed["route"],
                "retrieval_level": routed["retrieval_level"],
                "reason": routed["reason"],
                "retrieved_items": routed["retrieved_items"],
            }
        )
    router_df = pd.DataFrame(router_rows)
    router_df.to_csv(output_dir / "retrieval_router_demo.csv", index=False)

    hop2_rate = float(correction_df["hop2_sufficient"].mean()) if not correction_df.empty else 0.0
    final_rate = float(correction_df["final_sufficient"].mean()) if not correction_df.empty else 0.0
    hop2_chars = float(correction_df["hop2_context_chars"].mean()) if not correction_df.empty else 0.0
    final_chars = float(correction_df["final_context_chars"].mean()) if not correction_df.empty else 0.0
    llm_summary_rate = (
        float(community_reports["summary_method"].eq("LLM").mean())
        if not community_reports.empty and "summary_method" in community_reports
        else 0.0
    )
    before_after = pd.DataFrame(
        [
            {"bonus": "Self-Correction", "metric": "sufficiency_rate", "before_hop2": hop2_rate, "after_final": final_rate, "delta": final_rate - hop2_rate},
            {"bonus": "Self-Correction", "metric": "mean_context_chars", "before_hop2": hop2_chars, "after_final": final_chars, "delta": final_chars - hop2_chars},
            {"bonus": "Community Reports", "metric": "llm_summary_rate", "before_hop2": 0.0, "after_final": llm_summary_rate, "delta": llm_summary_rate},
        ]
    )
    before_after.to_csv(output_dir / "bonus_before_after.csv", index=False)

    route_counts = correction_df["route"].value_counts().to_dict() if not correction_df.empty else {}
    extra = pd.DataFrame(
        [
            {"bonus": "Global Search Community Reports", "metric": "llm_summaries", "value": int(community_reports.get("summary_method", pd.Series(dtype=str)).eq("LLM").sum())},
            {"bonus": "Global Search Community Reports", "metric": "llm_summary_rate", "value": llm_summary_rate},
            {"bonus": "Low-level / High-level Router", "metric": "global_routes", "value": int(router_df["route"].eq("global").sum())},
            {"bonus": "Low-level / High-level Router", "metric": "local_routes", "value": int(router_df["route"].eq("local").sum())},
            {"bonus": "Self-Correction LLM Sufficiency", "metric": "hop2", "value": int(route_counts.get("hop2", 0))},
            {"bonus": "Self-Correction LLM Sufficiency", "metric": "hop3", "value": int(route_counts.get("hop3", 0))},
            {"bonus": "Self-Correction LLM Sufficiency", "metric": "hop3+vector", "value": int(route_counts.get("hop3+vector", 0))},
            {"bonus": "Self-Correction LLM Sufficiency", "metric": "final_sufficient_rate", "value": final_rate},
        ]
    )
    combined = pd.concat([base, extra], ignore_index=True)
    combined.to_csv(output_dir / "bonus_metrics.csv", index=False)
    return combined


_impl.write_bonus_metrics = _write_bonus_metrics_strict

_original_generate_reports = _impl.generate_reports


def _generate_reports_with_bonus(*args: Any, **kwargs: Any) -> None:
    _original_generate_reports(*args, **kwargs)
    reports_dir = Path(args[7] if len(args) > 7 else kwargs["reports_dir"])
    output_dir = Path("outputs")
    for name in ["graphrag_eval_results.csv", "graphrag_vs_flatrag_summary.csv"]:
        source = output_dir / name
        if source.exists():
            shutil.copyfile(source, reports_dir / name)

    evidence_lines = ["\n## Strict Bonus Verification\n"]
    before_after_path = output_dir / "bonus_before_after.csv"
    router_path = output_dir / "retrieval_router_demo.csv"
    correction_path = output_dir / "self_correction_bonus_eval.csv"
    coref_path = output_dir / "coref_spotcheck.csv"
    if before_after_path.exists():
        before_after = pd.read_csv(before_after_path)
        for row in before_after.itertuples(index=False):
            evidence_lines.append(
                f"- {row.bonus} / {row.metric}: before={float(row.before_hop2):.3f}, after={float(row.after_final):.3f}, delta={float(row.delta):.3f}"
            )
    if router_path.exists():
        router = pd.read_csv(router_path)
        evidence_lines.append(
            f"- Query router demo: local={int(router.route.eq('local').sum())}, global={int(router.route.eq('global').sum())}."
        )
    if correction_path.exists():
        correction = pd.read_csv(correction_path)
        evidence_lines.append(
            f"- LLM self-correction sample={len(correction)}; vector fallbacks={int(correction.route.eq('hop3+vector').sum())}; every row has a terminal stop condition."
        )
    if coref_path.exists():
        evidence_lines.append("- Coreference spot-check exported to `outputs/coref_spotcheck.csv`.")
    evidence = "\n".join(evidence_lines) + "\n"
    for name in ["technical_defense.md", "reflection_ChuNguyenTuanAnh.md", "lab_report.md"]:
        path = reports_dir / name
        if path.exists():
            path.write_text(path.read_text(encoding="utf-8") + evidence, encoding="utf-8")


_impl.generate_reports = _generate_reports_with_bonus

# Re-export the patched instructor runtime. Existing notebook imports remain stable.
from lab19_runtime_impl import *  # noqa: E402,F401,F403
