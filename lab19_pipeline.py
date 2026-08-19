"""Lab 19 preprocessing and schema invariants.

Heavy external-service orchestration is added on top of these deterministic
building blocks. This layer deliberately enforces the assignment's first-5000
scope before any filtering, deduplication, chunking, or retrieval indexing.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

import pandas as pd

from lab19_core import chunk_words, normalize_text, validate_provenance_dataframe
from lab19_utils import FIRST_N_ROWS, select_stratified_indices, take_first_n_rows

SOURCE_MAX_ROWS = FIRST_N_ROWS
CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
EXTRACTION_MAX_CHUNKS = 400

ALLOWED_NODE_TYPES = {"Company", "Person", "Technology"}
ALLOWED_RELATIONS = {
    "ACQUIRED",
    "DEVELOPED",
    "INVESTED_IN",
    "FOUNDED",
    "WORKED_AT",
    "PARTNERED_WITH",
    "USES",
    "LEADS",
}

_COREF_RE = re.compile(
    r"\b(?:it|they|them|their|he|him|his|she|her|hers|the company|the startup)\b",
    flags=re.IGNORECASE,
)


def _sha1(value: object) -> str:
    return hashlib.sha1(str(value).encode("utf-8", errors="ignore")).hexdigest()


def _pick_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    lookup = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise KeyError(f"Missing one of columns: {list(candidates)}")
    return None


def standardize_news(raw: pd.DataFrame) -> pd.DataFrame:
    """Lock to rows 0..4999, normalize fields, then exact-deduplicate.

    The source scope is applied *before* all other operations. `source_row_id`
    always refers to the zero-based row position in that authoritative prefix.
    No sampling or shuffling is performed.
    """
    scoped = take_first_n_rows(raw, SOURCE_MAX_ROWS).copy()
    scoped["source_row_id"] = range(len(scoped))

    text_col = _pick_col(scoped, ["text", "content", "article", "body", "story"])
    title_col = _pick_col(scoped, ["title", "headline"], required=False)
    date_col = _pick_col(
        scoped,
        ["published_date", "date", "published_at", "created_at"],
        required=False,
    )
    id_col = _pick_col(scoped, ["id", "article_id", "story_id", "uuid"], required=False)

    out = pd.DataFrame(index=scoped.index)
    out["source_row_id"] = scoped["source_row_id"].astype(int)
    out["text"] = scoped[text_col].fillna("").map(normalize_text)
    out["title"] = (
        scoped[title_col].fillna("").map(normalize_text) if title_col else ""
    )

    if date_col:
        out["published_date"] = (
            pd.to_datetime(scoped[date_col], errors="coerce", utc=True)
            .dt.strftime("%Y-%m-%d")
            .fillna("")
        )
    else:
        out["published_date"] = ""

    if id_col:
        raw_ids = scoped[id_col].fillna("").astype(str).map(normalize_text)
        out["article_id"] = [
            rid if rid else _sha1(f"{row_id}:{title}\n{text}")[:20]
            for rid, row_id, title, text in zip(
                raw_ids,
                out["source_row_id"],
                out["title"],
                out["text"],
            )
        ]
    else:
        out["article_id"] = [
            _sha1(f"{row_id}:{title}\n{text}")[:20]
            for row_id, title, text in zip(
                out["source_row_id"], out["title"], out["text"]
            )
        ]

    out = out[out["text"].str.len() >= 80].copy()
    out["dedup_key"] = [
        _sha1(normalize_text(f"{title}\n{text}").lower())
        for title, text in zip(out["title"], out["text"])
    ]
    out = (
        out.drop_duplicates("dedup_key", keep="first")
        .drop(columns="dedup_key")
        .reset_index(drop=True)
    )
    return out


def build_chunks(
    news_df: pd.DataFrame,
    *,
    size: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> pd.DataFrame:
    """Chunk every standardized article; never truncate from the head."""
    required = {
        "source_row_id",
        "article_id",
        "title",
        "published_date",
        "text",
    }
    missing = required.difference(news_df.columns)
    if missing:
        raise ValueError(f"news dataframe missing columns: {sorted(missing)}")

    rows: list[dict] = []
    for row in news_df.itertuples(index=False):
        for idx, text in enumerate(chunk_words(row.text, size=size, overlap=overlap)):
            rows.append(
                {
                    "chunk_id": f"{row.article_id}::c{idx:04d}",
                    "source_row_id": int(row.source_row_id),
                    "article_id": row.article_id,
                    "title": row.title,
                    "published_date": row.published_date,
                    "text": text,
                }
            )
    return pd.DataFrame(rows)


def select_extraction_source(
    chunks_df: pd.DataFrame,
    *,
    limit: int = EXTRACTION_MAX_CHUNKS,
) -> pd.DataFrame:
    """Pick a deterministic corpus-wide extraction subset without gold leakage."""
    if chunks_df.empty:
        return chunks_df.copy()
    indices = select_stratified_indices(len(chunks_df), limit)
    return chunks_df.iloc[indices].copy().reset_index(drop=True)


def needs_coreference(text: str) -> bool:
    """Only spend an LLM coreference call when a conservative trigger exists."""
    return bool(_COREF_RE.search(normalize_text(text)))


def validate_extracted_triples(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "source_raw",
        "source_type",
        "relation",
        "target_raw",
        "target_type",
        "source_chunk_id",
        "published_date",
        "evidence",
        "confidence",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"triple dataframe missing columns: {sorted(missing)}")

    out = df.copy()
    if out.empty:
        return out

    bad_node_type = ~out["source_type"].isin(ALLOWED_NODE_TYPES) | ~out[
        "target_type"
    ].isin(ALLOWED_NODE_TYPES)
    if bool(bad_node_type.any()):
        raise ValueError(f"invalid node type rows: {int(bad_node_type.sum())}")

    bad_relation = ~out["relation"].isin(ALLOWED_RELATIONS)
    if bool(bad_relation.any()):
        raise ValueError(f"invalid relation rows: {int(bad_relation.sum())}")

    out["source_raw"] = out["source_raw"].map(normalize_text)
    out["target_raw"] = out["target_raw"].map(normalize_text)
    if bool(out["source_raw"].eq("").any() | out["target_raw"].eq("").any()):
        raise ValueError("blank entity mention in extracted triples")

    confidence = pd.to_numeric(out["confidence"], errors="coerce")
    if bool(confidence.isna().any() | (confidence < 0).any() | (confidence > 1).any()):
        raise ValueError("confidence must be numeric in [0, 1]")
    out["confidence"] = confidence.astype(float)

    validate_provenance_dataframe(out)
    return out.reset_index(drop=True)
