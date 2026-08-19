"""Deterministic core logic for Lab 19.

This module is deliberately API/DB-free so it can be exercised by lightweight
CI. Heavy Hugging Face, Groq, OpenAI and Neo4j orchestration lives elsewhere.
"""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
import re
import unicodedata
from typing import Iterable

import pandas as pd

CORP_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "ltd", "limited",
    "llc", "plc", "co", "company",
}
SUPER_NODE_DEGREE = 100
SUPER_NODE_EDGE_CAP = 50


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def chunk_words(text: str, size: int = 220, overlap: int = 40) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    words = normalize_text(text).split()
    if not words:
        return []
    step = size - overlap
    out: list[str] = []
    for start in range(0, len(words), step):
        part = words[start:start + size]
        if not part:
            break
        out.append(" ".join(part))
        if start + size >= len(words):
            break
    return out


def normalize_entity(name: object) -> str:
    text = normalize_text(name).lower()
    text = re.sub(r"[^\w\s\-.]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_corporate_suffix(name: object) -> str:
    tokens = normalize_entity(name).replace(".", "").split()
    while tokens and tokens[-1] in CORP_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def merge_guard(a: str, b: str, entity_type: str) -> bool:
    """Conservative lexical guard for entity resolution.

    It intentionally prefers false negatives over dangerous false merges.
    """
    na, nb = normalize_entity(a), normalize_entity(b)
    if not na or not nb:
        return False
    if na == nb:
        return True

    typ = str(entity_type or "").lower()
    if typ == "company":
        sa, sb = strip_corporate_suffix(a), strip_corporate_suffix(b)
        if sa and sa == sb:
            return True
        ta, tb = sa.split(), sb.split()
        # A company name being a strict prefix of a longer phrase is usually a
        # product/subsidiary mention (Apple vs Apple Watch), so reject it.
        if sa and sb and (sa.startswith(sb + " ") or sb.startswith(sa + " ")):
            return False
        return SequenceMatcher(None, sa, sb).ratio() >= 0.86 and abs(len(ta) - len(tb)) <= 1

    if typ == "person":
        ta, tb = na.split(), nb.split()
        if len(ta) < 2 or len(tb) < 2:
            return False
        # Matching surname alone is not enough; require same first and last token.
        if ta[0] != tb[0] or ta[-1] != tb[-1]:
            return False
        return SequenceMatcher(None, na, nb).ratio() >= 0.88

    # Technology/product names are semantically fragile. Require very strong
    # lexical evidence unless they normalize exactly.
    if na.startswith(nb + " ") or nb.startswith(na + " "):
        return False
    return SequenceMatcher(None, na, nb).ratio() >= 0.92


def _simhash_features(text: str) -> Iterable[str]:
    tokens = re.findall(r"[a-z0-9]+", normalize_text(text).lower())
    if len(tokens) < 3:
        return tokens
    return (" ".join(tokens[i:i + 3]) for i in range(len(tokens) - 2))


def simhash64(text: str) -> int:
    weights = [0] * 64
    seen = False
    for feature in _simhash_features(text):
        seen = True
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big", signed=False)
        for bit in range(64):
            weights[bit] += 1 if (value >> bit) & 1 else -1
    if not seen:
        return 0
    out = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            out |= 1 << bit
    return out


def hamming_distance(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


def near_dedup_dataframe(
    df: pd.DataFrame,
    *,
    max_hamming: int = 3,
    bands: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Near-deduplicate with SimHash LSH rather than O(N^2) pairwise scans."""
    required = {"title", "text", "source_row_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"near dedup missing columns: {sorted(missing)}")
    if max_hamming < 0:
        raise ValueError("max_hamming must be >= 0")
    if bands <= 0 or 64 % bands:
        raise ValueError("bands must divide 64")

    band_bits = 64 // bands
    band_mask = (1 << band_bits) - 1
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    kept_rows: list[int] = []
    kept_hashes: list[int] = []
    audit: list[dict] = []

    for pos, row in enumerate(df.itertuples(index=False)):
        combined = f"{getattr(row, 'title', '')}\n{getattr(row, 'text', '')}"
        value = simhash64(combined)
        candidates: set[int] = set()
        for band in range(bands):
            key = (band, (value >> (band * band_bits)) & band_mask)
            candidates.update(buckets.get(key, []))

        duplicate_of: int | None = None
        for kept_pos in sorted(candidates):
            if hamming_distance(value, kept_hashes[kept_pos]) <= max_hamming:
                duplicate_of = kept_pos
                break

        if duplicate_of is None:
            kept_slot = len(kept_rows)
            kept_rows.append(pos)
            kept_hashes.append(value)
            for band in range(bands):
                key = (band, (value >> (band * band_bits)) & band_mask)
                buckets[key].append(kept_slot)
            continue

        kept_df_pos = kept_rows[duplicate_of]
        kept = df.iloc[kept_df_pos]
        dropped = df.iloc[pos]
        audit.append(
            {
                "kept_source_row_id": int(kept["source_row_id"]),
                "dropped_source_row_id": int(dropped["source_row_id"]),
                "kept_simhash": int(kept_hashes[duplicate_of]),
                "dropped_simhash": int(value),
                "hamming_distance": hamming_distance(value, kept_hashes[duplicate_of]),
                "decision": "DROP_NEAR_DUP",
            }
        )

    out = df.iloc[kept_rows].copy().reset_index(drop=True)
    return out, pd.DataFrame(audit)


def supernode_edge_limit(
    degree: int,
    *,
    requested: int,
    threshold: int = SUPER_NODE_DEGREE,
    cap: int = SUPER_NODE_EDGE_CAP,
) -> int:
    if degree < 0 or requested < 0:
        raise ValueError("degree/requested must be non-negative")
    return min(requested, cap) if degree > threshold else requested


def validate_provenance_dataframe(df: pd.DataFrame) -> None:
    required = ["source_chunk_id", "published_date", "evidence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"provenance missing columns: {missing}")
    if df.empty:
        return
    invalid = pd.Series(False, index=df.index)
    for col in required:
        invalid |= df[col].fillna("").astype(str).str.strip().eq("")
    if bool(invalid.any()):
        raise ValueError(f"provenance invalid rows: {int(invalid.sum())}")
