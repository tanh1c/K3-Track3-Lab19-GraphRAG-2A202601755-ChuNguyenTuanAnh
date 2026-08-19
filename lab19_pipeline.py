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
from lab19_utils import FIRST_N_ROWS, take_first_n_rows

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

_RELATION_SIGNAL_PATTERNS = (
    re.compile(r"\bacquir(?:e|es|ed|ing|er|ers|isition|isitions)\b", re.I),
    re.compile(r"\bdevelop(?:s|ed|ing|er|ers|ment|ments)?\b", re.I),
    re.compile(r"\binvest(?:s|ed|ing|ment|ments|or|ors)?\b", re.I),
    re.compile(r"\bfound(?:s|ed|er|ers|ing)?\b", re.I),
    re.compile(r"\bwork(?:s|ed|ing)?\s+(?:at|for)\b", re.I),
    re.compile(r"\bpartner(?:s|ed|ing|ship|ships)?\b", re.I),
    re.compile(r"\buses?\b|\busing\b|\bused\b", re.I),
    re.compile(r"\blead(?:s|ing)?\b|\bled\b", re.I),
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
    HackerNoon's current dataset schema exposes article snippets as `description`;
    for that schema the retrieval text is `title + description` so headline facts
    are not discarded. No sampling or shuffling is performed.
    """
    scoped = take_first_n_rows(raw, SOURCE_MAX_ROWS).copy()
    scoped["source_row_id"] = range(len(scoped))

    text_col = _pick_col(
        scoped,
        ["text", "content", "article", "body", "story", "description"],
    )
    title_col = _pick_col(scoped, ["title", "headline"], required=False)
    date_col = _pick_col(
        scoped,
        ["published_date", "date", "published_at", "created_at"],
        required=False,
    )
    id_col = _pick_col(
        scoped,
        ["id", "article_id", "story_id", "uuid", "url"],
        required=False,
    )

    out = pd.DataFrame(index=scoped.index)
    out["source_row_id"] = scoped["source_row_id"].astype(int)
    titles = (
        scoped[title_col].fillna("").map(normalize_text)
        if title_col
        else pd.Series("", index=scoped.index, dtype="object")
    )
    bodies = scoped[text_col].fillna("").map(normalize_text)
    out["title"] = titles
    if str(text_col).lower() == "description":
        out["text"] = [
            normalize_text(f"{title}. {body}" if title and body else title or body)
            for title, body in zip(titles, bodies)
        ]
    else:
        out["text"] = bodies

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


def _relation_signal_score(text: object) -> int:
    """Count schema-derived relation cues in a chunk."""
    normalized = normalize_text(text)
    return sum(min(2, len(pattern.findall(normalized))) for pattern in _RELATION_SIGNAL_PATTERNS)


def select_extraction_source(
    chunks_df: pd.DataFrame,
    *,
    limit: int = EXTRACTION_MAX_CHUNKS,
) -> pd.DataFrame:
    """Pick a deterministic, corpus-wide, relation-rich extraction subset."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if chunks_df.empty:
        return chunks_df.copy()
    if "text" not in chunks_df.columns:
        raise ValueError("chunks dataframe must contain text")

    total = len(chunks_df)
    if limit >= total:
        return chunks_df.copy().reset_index(drop=True)
    if limit == 1:
        return chunks_df.iloc[[0]].copy().reset_index(drop=True)
    if limit == 2:
        return chunks_df.iloc[[0, total - 1]].copy().reset_index(drop=True)

    selected = [0]
    interior_slots = limit - 2
    interior_count = total - 2

    for slot in range(interior_slots):
        start = 1 + (slot * interior_count) // interior_slots
        end_exclusive = 1 + ((slot + 1) * interior_count) // interior_slots
        if end_exclusive <= start:
            end_exclusive = start + 1
        candidates = list(range(start, min(end_exclusive, total - 1)))
        if not candidates:
            continue

        center = (candidates[0] + candidates[-1]) / 2.0
        best = min(
            candidates,
            key=lambda idx: (
                -_relation_signal_score(chunks_df.iloc[idx]["text"]),
                abs(idx - center),
                idx,
            ),
        )
        selected.append(best)

    selected.append(total - 1)
    selected = sorted(set(selected))
    if len(selected) != limit:
        raise AssertionError(
            f"extraction coverage selector expected {limit} unique chunks, got {len(selected)}"
        )
    return chunks_df.iloc[selected].copy().reset_index(drop=True)


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
