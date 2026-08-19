import pandas as pd
import pytest

from lab19_core import (
    chunk_words,
    hamming_distance,
    merge_guard,
    near_dedup_dataframe,
    normalize_text,
    supernode_edge_limit,
    validate_provenance_dataframe,
)


def test_normalize_text_nfkc_and_whitespace():
    assert normalize_text("  AI\u3000  platform\n") == "AI platform"


def test_chunk_words_uses_overlap_without_dropping_tail():
    text = " ".join(f"w{i}" for i in range(12))
    chunks = chunk_words(text, size=5, overlap=2)
    assert chunks == [
        "w0 w1 w2 w3 w4",
        "w3 w4 w5 w6 w7",
        "w6 w7 w8 w9 w10",
        "w9 w10 w11",
    ]


def test_entity_guard_allows_corporate_suffix_but_blocks_product_and_people_false_merge():
    assert merge_guard("Microsoft Corp.", "Microsoft", "Company") is True
    assert merge_guard("Apple", "Apple Watch", "Company") is False
    assert merge_guard("Sam Altman", "Steve Altman", "Person") is False


def test_near_dedup_removes_small_rewrite_and_keeps_unrelated_article():
    df = pd.DataFrame(
        {
            "title": ["A", "A repost", "B"],
            "text": [
                "Microsoft announced a new cloud AI platform for enterprise customers today.",
                "Microsoft announced a new cloud AI platform for enterprise customers today!",
                "Aeris acquired Ericsson IoT assets and expanded connectivity globally.",
            ],
            "source_row_id": [0, 1, 2],
        }
    )
    out, audit = near_dedup_dataframe(df, max_hamming=3)
    assert out.source_row_id.tolist() == [0, 2]
    assert len(audit) == 1
    assert audit.iloc[0].kept_source_row_id == 0
    assert audit.iloc[0].dropped_source_row_id == 1
    assert hamming_distance(int(audit.iloc[0].kept_simhash), int(audit.iloc[0].dropped_simhash)) <= 3


def test_supernode_policy_caps_high_degree_nodes():
    assert supernode_edge_limit(99, requested=80) == 80
    assert supernode_edge_limit(101, requested=80) == 50
    assert supernode_edge_limit(101, requested=20) == 20


def test_provenance_validator_rejects_missing_required_fields():
    good = pd.DataFrame(
        [{"source_chunk_id": "c1", "published_date": "2023-01-01", "evidence": "x"}]
    )
    validate_provenance_dataframe(good)

    bad = pd.DataFrame(
        [{"source_chunk_id": "", "published_date": "2023-01-01", "evidence": "x"}]
    )
    with pytest.raises(ValueError, match="provenance"):
        validate_provenance_dataframe(bad)
