import json
from pathlib import Path


NOTEBOOK = Path("Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb")


def notebook_text() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    return "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
        for cell in payload["cells"]
    )


def test_notebook_is_executable_entrypoint_for_ci_and_colab():
    text = notebook_text()
    assert "LAB_RUN_MODE" in text
    assert "run_lab(RUN_MODE)" in text
    assert "FIRST_5000_ROWS_ONLY" in text


def test_notebook_uses_uploaded_canonical_golden_dataset():
    text = notebook_text()
    assert "data/graphrag_golden_50_first5000.csv" in text
    assert "validate_golden_dataset" in text


def test_notebook_surfaces_rubric_and_bonus_outputs():
    text = notebook_text()
    for required in [
        "graphrag_eval_results.csv",
        "graphrag_vs_flatrag_summary.csv",
        "entity_resolution_audit.csv",
        "supernode_diagnostics.csv",
        "community_reports.csv",
        "bonus_metrics.csv",
    ]:
        assert required in text
