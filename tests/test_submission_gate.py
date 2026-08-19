import json
from pathlib import Path

import pandas as pd
import pytest

from lab19_submission import validate_submission


def write_minimal_valid_submission(root: Path, *, golden_rows: int = 50) -> None:
    outputs = root / "outputs"
    reports = root / "reports"
    outputs.mkdir(parents=True)
    reports.mkdir(parents=True)

    groups = ["factoid", "multi-hop", "cross-doc"]
    eval_rows = []
    for i in range(golden_rows):
        eval_rows.append(
            {
                "id": f"G{i:02d}",
                "group": groups[i % 3],
                "question": f"q{i}",
                "reference_answer": f"r{i}",
                "flat_answer": "flat",
                "graph_answer": "graph",
                "flat_comprehensiveness": 3,
                "graph_comprehensiveness": 4,
                "flat_faithfulness": 4,
                "graph_faithfulness": 4,
                "flat_multi_hop_reasoning": 2,
                "graph_multi_hop_reasoning": 4,
                "flat_latency_s": 1.0,
                "graph_latency_s": 2.0,
                "flat_total_tokens": 100,
                "graph_total_tokens": 160,
                "flat_judge_rationale": "evidence",
                "graph_judge_rationale": "evidence",
            }
        )
    pd.DataFrame(eval_rows).to_csv(outputs / "graphrag_eval_results.csv", index=False)

    pd.DataFrame(
        [
            {
                "question_group": "multi-hop",
                "metric": "Comprehensiveness",
                "flat_rag": 3.0,
                "graph_rag": 4.0,
                "delta_graph_minus_flat": 1.0,
            }
        ]
    ).to_csv(outputs / "graphrag_vs_flatrag_summary.csv", index=False)

    pd.DataFrame(
        [
            {"type": "Company", "left": "Apple", "right": "Apple Watch", "similarity": 0.9, "decision": "REJECT_GUARD"},
            {"type": "Company", "left": "MSFT", "right": "Microsoft", "similarity": 1.0, "decision": "MERGE_MANUAL"},
        ]
        * 5
    ).to_csv(outputs / "entity_resolution_audit.csv", index=False)
    pd.DataFrame([{"id": "n1", "name": "Microsoft", "type": "Company", "degree": 120}]).to_csv(
        outputs / "supernode_diagnostics.csv", index=False
    )
    pd.DataFrame([{"community_id": 0, "size": 10, "hubs": "A", "dominant_relations": "USES", "report": "community"}]).to_csv(
        outputs / "community_reports.csv", index=False
    )
    pd.DataFrame(
        [
            {"bonus": "Near-Dedup SimHash/LSH", "metric": "near_duplicates_removed", "value": 2},
            {"bonus": "Global Search Community Reports", "metric": "communities", "value": 1},
            {"bonus": "Self-Correction Graph Retrieval", "metric": "hop2", "value": 1},
        ]
    ).to_csv(outputs / "bonus_metrics.csv", index=False)
    (outputs / "graph_checks.json").write_text(
        json.dumps({"nodes": 10, "edges": 20, "invalid_provenance_edges": 0}), encoding="utf-8"
    )
    (outputs / "run_manifest.json").write_text(
        json.dumps(
            {
                "source_policy": "FIRST_5000_ROWS_ONLY",
                "source_rows_downloaded": 5000,
                "golden_questions_evaluated": golden_rows,
                "invalid_provenance_edges": 0,
            }
        ),
        encoding="utf-8",
    )

    for name in [
        "lab_report.md",
        "technical_defense.md",
        "failure_analysis.md",
        "reflection_ChuNguyenTuanAnh.md",
    ]:
        (reports / name).write_text(
            "# Complete report\nEmpirical evidence with concrete threshold 0.90, provenance and failure analysis.\n",
            encoding="utf-8",
        )


def test_full_submission_gate_accepts_complete_50_question_artifacts(tmp_path):
    write_minimal_valid_submission(tmp_path)
    result = validate_submission(tmp_path, mode="full")
    assert result["ok"] is True
    assert result["golden_questions"] == 50
    assert result["invalid_provenance_edges"] == 0


def test_full_submission_gate_rejects_partial_evaluation(tmp_path):
    write_minimal_valid_submission(tmp_path, golden_rows=3)
    with pytest.raises(ValueError, match="50"):
        validate_submission(tmp_path, mode="full")


def test_submission_gate_rejects_report_placeholders(tmp_path):
    write_minimal_valid_submission(tmp_path)
    report = tmp_path / "reports" / "technical_defense.md"
    report.write_text("# TODO\n[Họ và Tên]\nTO_BE_FILLED_FROM_DATASET", encoding="utf-8")
    with pytest.raises(ValueError, match="placeholder"):
        validate_submission(tmp_path, mode="full")


def test_submission_gate_rejects_nonzero_provenance_failures(tmp_path):
    write_minimal_valid_submission(tmp_path)
    checks = tmp_path / "outputs" / "graph_checks.json"
    checks.write_text(json.dumps({"invalid_provenance_edges": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        validate_submission(tmp_path, mode="full")
