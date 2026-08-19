"""Runtime guardrails for Lab 19.

The instructor implementation remains in :mod:`lab19_runtime_impl`. This wrapper
keeps that code inspectable while making OpenAI the primary LLM provider for
coreference, NER/RE extraction, answer synthesis and evaluation support. Groq is
optional only and is not required on the critical path.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path
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

    def _pace(self) -> None:
        # Deliberately no fixed sleep. The OpenAI SDK handles transient 429/5xx
        # retries with provider-aware backoff; calls remain sequential in this lab.
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
# instructor's provider class globally. All run_lab references resolve this symbol
# at call time, so the notebook now instantiates OpenAIRuntime.
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


def _resilient_extract_triples(
    source_df: pd.DataFrame,
    llm: Any,
    config: Any,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extraction with resumable checkpoints; failed batches are never marked done."""
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
        prompt = (
            "Return {\"items\":[{\"chunk_id\":\"...\",\"relations\":[{"
            "\"source\":\"...\",\"source_type\":\"Company|Person|Technology\","
            "\"relation\":\"ALLOWED_RELATION\",\"target\":\"...\","
            "\"target_type\":\"Company|Person|Technology\",\"evidence\":\"...\","
            "\"confidence\":0.0}]}]}. Include one items entry for EVERY input chunk, even when relations is empty. INPUT:\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        try:
            obj, _ = llm.chat_json(system=system, user=prompt, model=llm.fast_model, max_tokens=1800)
            if not valid_extraction_payload(obj, expected_ids):
                raise ValueError("Malformed or incomplete extraction items envelope")
        except Exception as exc:
            errors.append(
                {
                    "start": int(start),
                    "chunk_ids": json.dumps(sorted(expected_ids)),
                    "provider": "openai",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[extract] OpenAI batch incomplete and NOT checkpointed: {type(exc).__name__}: {exc}")
            continue

        by_id = {str(item.get("chunk_id")): item for item in obj.get("items", [])}
        for r in batch.itertuples(index=False):
            checkpoint.upsert(by_id[str(r.chunk_id)])

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


_impl.extract_triples = _resilient_extract_triples

# Re-export the patched instructor runtime. Because _impl.GroqRuntime was replaced
# above, existing notebook code can keep its old imports while actually using OpenAI.
from lab19_runtime_impl import *  # noqa: E402,F401,F403
