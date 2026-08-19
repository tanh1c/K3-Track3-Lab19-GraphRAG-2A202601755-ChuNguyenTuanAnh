"""Shared utilities for Lab 19 CI and notebook execution.

The module intentionally contains deterministic, testable logic so the
lightweight CI never needs Neo4j, Hugging Face, Groq, or OpenAI credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

FIRST_N_ROWS = 5000
GOLDEN_GROUPS = {"factoid", "multi-hop", "cross-doc"}
GOLDEN_REQUIRED_COLUMNS = {
    "id",
    "group",
    "question",
    "reference_answer",
    "reference_evidence",
}


def take_first_n_rows(df: pd.DataFrame, n: int = FIRST_N_ROWS) -> pd.DataFrame:
    """Return the first *n* rows without sampling or reordering."""
    if n <= 0:
        raise ValueError("n must be positive")
    return df.iloc[:n].copy().reset_index(drop=True)


def select_stratified_indices(total: int, limit: int) -> list[int]:
    """Select deterministic, corpus-wide indices without benchmark leakage.

    When `limit < total`, the first and last records are always represented and
    the remaining points are evenly spaced over the full corpus.  This avoids
    the scaffold's head-only extraction bias while remaining query/gold agnostic.
    """
    if total < 0:
        raise ValueError("total must be >= 0")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if total == 0:
        return []
    if limit >= total:
        return list(range(total))
    if limit == 1:
        return [0]

    denom = limit - 1
    last = total - 1
    idx = [round(i * last / denom) for i in range(limit)]
    # For total > limit the spacing is >= 1, but keep this defensive assertion
    # so future changes cannot silently introduce duplicate work.
    if len(set(idx)) != len(idx):
        raise AssertionError("stratified selection produced duplicate indices")
    return idx


@dataclass(frozen=True)
class RateLimitPolicy:
    """Retry-delay policy used by the Groq wrapper.

    `retry_after_s` always wins when the provider supplies it. Otherwise the
    policy uses capped exponential backoff with bounded jitter.
    """

    base_delay_s: float = 2.0
    max_delay_s: float = 90.0
    jitter_s: float = 1.0

    def delay_for(
        self,
        attempt: int,
        retry_after_s: float | None = None,
        *,
        rng: random.Random | None = None,
    ) -> float:
        if attempt < 0:
            raise ValueError("attempt must be >= 0")
        if retry_after_s is not None and retry_after_s >= 0:
            return min(float(retry_after_s), self.max_delay_s)
        delay = min(self.base_delay_s * (2**attempt), self.max_delay_s)
        if self.jitter_s > 0:
            rng = rng or random
            delay = min(delay + rng.uniform(0.0, self.jitter_s), self.max_delay_s)
        return float(delay)


def parse_retry_after_seconds(headers: Mapping[str, Any] | None) -> float | None:
    """Read Groq's numeric Retry-After header case-insensitively."""
    if not headers:
        return None
    for key, value in headers.items():
        if str(key).lower() != "retry-after":
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, parsed)
    return None


class JsonlCheckpoint:
    """Append-only keyed JSONL checkpoint with last-write-wins semantics."""

    def __init__(self, path: str | Path, key_field: str):
        if not key_field:
            raise ValueError("key_field is required")
        self.path = Path(path)
        self.key_field = key_field

    def _resolved(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return out
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed checkpoint JSON at {self.path}:{line_no}"
                    ) from exc
                if self.key_field not in row:
                    raise ValueError(
                        f"Checkpoint row missing key field {self.key_field!r}"
                    )
                out[str(row[self.key_field])] = row
        return out

    def completed_keys(self) -> set[str]:
        return set(self._resolved())

    def rows(self) -> list[dict[str, Any]]:
        return list(self._resolved().values())

    def upsert(self, row: Mapping[str, Any]) -> None:
        if self.key_field not in row:
            raise ValueError(f"row missing key field {self.key_field!r}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            f.write("\n")
            f.flush()


def load_golden_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _blank_count(series: pd.Series) -> int:
    return int(series.fillna("").astype(str).str.strip().eq("").sum())


def validate_golden_dataset(df: pd.DataFrame) -> dict[str, Any]:
    missing_cols = sorted(GOLDEN_REQUIRED_COLUMNS.difference(df.columns))
    if missing_cols:
        raise ValueError(f"Missing golden columns: {missing_cols}")

    missing_answers = _blank_count(df["reference_answer"])
    if missing_answers:
        raise ValueError(f"Golden dataset has {missing_answers} blank reference_answer values")

    missing_evidence = _blank_count(df["reference_evidence"])
    if missing_evidence:
        raise ValueError(f"Golden dataset has {missing_evidence} blank reference_evidence values")

    blank_questions = _blank_count(df["question"])
    if blank_questions:
        raise ValueError(f"Golden dataset has {blank_questions} blank question values")

    groups = sorted(set(df["group"].dropna().astype(str).str.strip()))
    unknown_groups = sorted(set(groups).difference(GOLDEN_GROUPS))
    if unknown_groups:
        raise ValueError(f"Unknown golden groups: {unknown_groups}")
    missing_groups = sorted(GOLDEN_GROUPS.difference(groups))
    if missing_groups:
        raise ValueError(f"Golden dataset is missing groups: {missing_groups}")

    duplicate_ids = int(df["id"].duplicated().sum())
    if duplicate_ids:
        raise ValueError(f"Golden dataset has {duplicate_ids} duplicate ids")

    return {
        "rows": int(len(df)),
        "groups": groups,
        "missing_reference_answers": missing_answers,
        "missing_reference_evidence": missing_evidence,
        "duplicate_ids": duplicate_ids,
    }
