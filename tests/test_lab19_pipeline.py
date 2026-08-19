import pandas as pd
import pytest

from lab19_pipeline import (
    ALLOWED_NODE_TYPES,
    ALLOWED_RELATIONS,
    SOURCE_MAX_ROWS,
    build_chunks,
    needs_coreference,
    select_extraction_source,
    standardize_news,
    validate_extracted_triples,
)


def test_source_scope_is_locked_to_first_5000_rows_and_preserves_source_ids():
    raw = pd.DataFrame(
        {
            "title": [f"title {i}" for i in range(5100)],
            "text": [f"article body long enough for row {i} " * 4 for i in range(5100)],
            "published_date": ["2023-01-01"] * 5100,
        }
    )
    news = standardize_news(raw)
    assert SOURCE_MAX_ROWS == 5000
    assert len(news) == 5000
    assert news.iloc[0].source_row_id == 0
    assert news.iloc[-1].source_row_id == 4999


def test_build_chunks_keeps_source_row_id_and_does_not_head_truncate():
    news = pd.DataFrame(
        [
            {
                "source_row_id": 4999,
                "article_id": "last",
                "title": "last title",
                "published_date": "2023-01-02",
                "text": " ".join(f"w{i}" for i in range(15)),
            }
        ]
    )
    chunks = build_chunks(news, size=8, overlap=2)
    assert len(chunks) == 3
    assert chunks.source_row_id.tolist() == [4999, 4999, 4999]
    assert chunks.iloc[-1].chunk_id == "last::c0002"


def test_extraction_source_is_query_agnostic_and_spans_whole_chunk_table():
    chunks = pd.DataFrame(
        {
            "chunk_id": [f"c{i}" for i in range(5000)],
            "text": [f"text {i}" for i in range(5000)],
            "source_row_id": range(5000),
        }
    )
    selected = select_extraction_source(chunks, limit=400)
    assert len(selected) == 400
    assert selected.iloc[0].chunk_id == "c0"
    assert selected.iloc[-1].chunk_id == "c4999"
    assert selected.source_row_id.nunique() == 400


def test_extraction_source_prefers_relation_rich_chunks_inside_coverage_bins():
    chunks = pd.DataFrame(
        {
            "chunk_id": [f"c{i}" for i in range(12)],
            "source_row_id": range(12),
            "text": [
                "plain background text",
                "plain background text",
                "plain background text",
                "plain background text",
                "Microsoft partnered with Contoso and invested in its AI platform.",
                "plain background text",
                "plain background text",
                "Aeris acquired a technology business and uses its IoT platform.",
                "plain background text",
                "plain background text",
                "plain background text",
                "plain background text",
            ],
        }
    )
    selected = select_extraction_source(chunks, limit=4)
    assert selected.chunk_id.tolist() == ["c0", "c4", "c7", "c11"]


def test_coreference_filter_is_conservative_and_case_insensitive():
    assert needs_coreference("Microsoft launched it after testing.") is True
    assert needs_coreference("THE COMPANY announced a product.") is True
    assert needs_coreference("Microsoft announced Azure AI Studio.") is False


def test_allowed_schema_matches_assignment_allowlist():
    assert ALLOWED_NODE_TYPES == {"Company", "Person", "Technology"}
    assert ALLOWED_RELATIONS == {
        "ACQUIRED",
        "DEVELOPED",
        "INVESTED_IN",
        "FOUNDED",
        "WORKED_AT",
        "PARTNERED_WITH",
        "USES",
        "LEADS",
    }


def test_validate_extracted_triples_rejects_invalid_schema_and_missing_provenance():
    good = pd.DataFrame(
        [
            {
                "source_raw": "Microsoft",
                "source_type": "Company",
                "relation": "DEVELOPED",
                "target_raw": "Copilot",
                "target_type": "Technology",
                "source_chunk_id": "a::c0000",
                "published_date": "2023-03-01",
                "evidence": "Microsoft developed Copilot.",
                "confidence": 0.95,
            }
        ]
    )
    out = validate_extracted_triples(good)
    assert len(out) == 1

    invalid_relation = good.copy()
    invalid_relation.loc[0, "relation"] = "TRANSFERRED_TO"
    with pytest.raises(ValueError, match="relation"):
        validate_extracted_triples(invalid_relation)

    missing_provenance = good.copy()
    missing_provenance.loc[0, "source_chunk_id"] = ""
    with pytest.raises(ValueError, match="provenance"):
        validate_extracted_triples(missing_provenance)
