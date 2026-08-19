"""Machine-checkable submission gate for Lab 19.

The gate validates evidence produced by a real run. It never fabricates scores
or fills missing artifacts; failures are explicit so the notebook/CI can be
fixed before submission.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pandas as pd


REQUIRED_REPORTS = (
    "lab_report.md",
    "technical_defense.md",
    "failure_analysis.md",
    "reflection_ChuNguyenTuanAnh.md",
)
REQUIRED_OUTPUTS = (
    "graphrag_eval_results.csv",
    "graphrag_vs_flatrag_summary.csv",
    "entity_resolution_audit.csv",
    "supernode_diagnostics.csv",
    "community_reports.csv",
    "bonus_metrics.csv",
    "graph_checks.json",
    "run_manifest.json",
)
REQUIRED_EVAL_COLUMNS = {
    "id",
    "group",
    "question",
    "reference_answer",
    "flat_answer",
    "graph_answer",
    "flat_comprehensiveness",
    "graph_comprehensiveness",
    "flat_faithfulness",
    "graph_faithfulness",
    "flat_multi_hop_reasoning",
    "graph_multi_hop_reasoning",
    "flat_latency_s",
    "graph_latency_s",
    "flat_total_tokens",
    "graph_total_tokens",
    "flat_judge_rationale",
    "graph_judge_rationale",
}
GOLDEN_GROUPS = {"factoid", "multi-hop", "cross-doc"}
PLACEHOLDER_PATTERNS = (
    r"TO_BE_FILLED",
    r"\[Họ\s*(?:và|&)\s*Tên\]",
    r"\[Ngày/Tháng/Năm\]",
    r"\*Trả lời:\*\s*(?:\n\s*)?\[",
    r"\bTBD\b",
)


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"required artifact missing or empty: {path}")


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _blank_count(series: pd.Series) -> int:
    return int(series.fillna("").astype(str).str.strip().eq("").sum())


def _validate_reports(reports_dir: Path) -> None:
    for name in REQUIRED_REPORTS:
        path = reports_dir / name
        _require_file(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in PLACEHOLDER_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
                raise ValueError(f"report placeholder remains in {name}: {pattern}")


def validate_submission(root: str | Path = ".", *, mode: str = "full") -> dict[str, Any]:
    """Validate rubric-facing outputs and return a compact evidence summary.

    `full` requires all 50 golden questions. `smoke` accepts a partial benchmark
    but still enforces source scope, schemas, provenance, reports and bonus files.
    """
    root = Path(root)
    mode = str(mode).strip().lower()
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'")

    outputs = root / "outputs"
    reports = root / "reports"
    for name in REQUIRED_OUTPUTS:
        _require_file(outputs / name)
    _validate_reports(reports)

    eval_df = pd.read_csv(outputs / "graphrag_eval_results.csv")
    _require_columns(eval_df, REQUIRED_EVAL_COLUMNS, "evaluation")
    expected_questions = 50 if mode == "full" else 1
    if len(eval_df) < expected_questions:
        raise ValueError(
            f"{mode} submission must evaluate at least {expected_questions} golden questions; "
            f"found {len(eval_df)}"
        )
    if mode == "full" and len(eval_df) != 50:
        raise ValueError(f"full submission must evaluate exactly 50 golden questions; found {len(eval_df)}")
    if int(eval_df["id"].duplicated().sum()):
        raise ValueError("evaluation contains duplicate golden ids")
    if _blank_count(eval_df["reference_answer"]):
        raise ValueError("evaluation contains blank reference answers")
    if _blank_count(eval_df["flat_answer"]) or _blank_count(eval_df["graph_answer"]):
        raise ValueError("evaluation contains blank model answers")
    if _blank_count(eval_df["flat_judge_rationale"]) or _blank_count(eval_df["graph_judge_rationale"]):
        raise ValueError("evaluation contains blank judge rationale")
    groups = set(eval_df["group"].dropna().astype(str).str.strip())
    if mode == "full" and groups != GOLDEN_GROUPS:
        raise ValueError(f"full evaluation must cover {sorted(GOLDEN_GROUPS)}; found {sorted(groups)}")

    score_columns = [
        "flat_comprehensiveness",
        "graph_comprehensiveness",
        "flat_faithfulness",
        "graph_faithfulness",
        "flat_multi_hop_reasoning",
        "graph_multi_hop_reasoning",
    ]
    for column in score_columns:
        scores = pd.to_numeric(eval_df[column], errors="coerce")
        if bool(scores.isna().any() | (scores < 1).any() | (scores > 5).any()):
            raise ValueError(f"judge score outside 1..5 in {column}")

    graph_checks = json.loads((outputs / "graph_checks.json").read_text(encoding="utf-8"))
    invalid_provenance = int(graph_checks.get("invalid_provenance_edges", -1))
    if invalid_provenance != 0:
        raise ValueError(f"provenance integrity failed: invalid_provenance_edges={invalid_provenance}")

    manifest = json.loads((outputs / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_policy") != "FIRST_5000_ROWS_ONLY":
        raise ValueError("run manifest does not enforce FIRST_5000_ROWS_ONLY")
    if int(manifest.get("source_rows_downloaded", -1)) != 5000:
        raise ValueError("run manifest must prove exactly 5000 source rows")
    if int(manifest.get("invalid_provenance_edges", -1)) != 0:
        raise ValueError("run manifest reports provenance failures")
    if mode == "full" and int(manifest.get("golden_questions_evaluated", -1)) != 50:
        raise ValueError("full run manifest must prove all 50 golden questions were evaluated")

    audit_df = pd.read_csv(outputs / "entity_resolution_audit.csv")
    _require_columns(audit_df, {"decision", "similarity"}, "entity resolution audit")
    if len(audit_df) < 10:
        raise ValueError(f"entity resolution audit requires at least 10 rows; found {len(audit_df)}")
    allowed_decisions = {"MERGE_MANUAL", "MERGE_VECTOR", "REJECT_GUARD"}
    unknown = set(audit_df["decision"].dropna().astype(str)).difference(allowed_decisions)
    if unknown:
        raise ValueError(f"entity resolution audit has unknown decisions: {sorted(unknown)}")

    summary_df = pd.read_csv(outputs / "graphrag_vs_flatrag_summary.csv")
    if summary_df.empty:
        raise ValueError("comparison summary is empty")

    supernode_df = pd.read_csv(outputs / "supernode_diagnostics.csv")
    _require_columns(supernode_df, {"name", "degree"}, "supernode diagnostics")

    community_df = pd.read_csv(outputs / "community_reports.csv")
    if community_df.empty:
        raise ValueError("community reports are empty; Global Search bonus evidence missing")

    bonus_df = pd.read_csv(outputs / "bonus_metrics.csv")
    _require_columns(bonus_df, {"bonus", "metric", "value"}, "bonus metrics")
    bonus_names = set(bonus_df["bonus"].dropna().astype(str))
    required_bonus = {
        "Near-Dedup SimHash/LSH",
        "Global Search Community Reports",
        "Self-Correction Graph Retrieval",
    }
    missing_bonus = sorted(required_bonus.difference(bonus_names))
    if missing_bonus:
        raise ValueError(f"bonus evidence missing: {missing_bonus}")

    return {
        "ok": True,
        "mode": mode,
        "golden_questions": int(len(eval_df)),
        "groups": sorted(groups),
        "invalid_provenance_edges": invalid_provenance,
        "entity_audit_rows": int(len(audit_df)),
        "community_reports": int(len(community_df)),
        "bonus_evidence": sorted(required_bonus),
    }
