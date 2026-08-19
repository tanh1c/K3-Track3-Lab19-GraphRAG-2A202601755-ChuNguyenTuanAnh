import pandas as pd
import pytest

from lab19_utils import (
    FIRST_N_ROWS,
    JsonlCheckpoint,
    RateLimitPolicy,
    load_golden_dataset,
    parse_retry_after_seconds,
    select_stratified_indices,
    take_first_n_rows,
    validate_golden_dataset,
)


def test_first_n_policy_is_exactly_5000_and_preserves_order():
    df = pd.DataFrame({"row_id": range(6000), "value": range(6000)})
    out = take_first_n_rows(df)
    assert FIRST_N_ROWS == 5000
    assert len(out) == 5000
    assert out.iloc[0].row_id == 0
    assert out.iloc[-1].row_id == 4999


def test_first_n_policy_does_not_shuffle_short_inputs():
    df = pd.DataFrame({"row_id": [3, 1, 2]})
    out = take_first_n_rows(df)
    assert out.row_id.tolist() == [3, 1, 2]


def test_rate_limit_policy_uses_retry_after_when_present():
    policy = RateLimitPolicy(base_delay_s=2.0, max_delay_s=90.0, jitter_s=0.0)
    assert policy.delay_for(attempt=3, retry_after_s=17.5) == 17.5


def test_rate_limit_policy_exponential_backoff_is_capped():
    policy = RateLimitPolicy(base_delay_s=2.0, max_delay_s=30.0, jitter_s=0.0)
    assert policy.delay_for(attempt=0) == 2.0
    assert policy.delay_for(attempt=3) == 16.0
    assert policy.delay_for(attempt=10) == 30.0


def test_retry_after_parser_accepts_numeric_header():
    assert parse_retry_after_seconds({"retry-after": "12.5"}) == 12.5
    assert parse_retry_after_seconds({"Retry-After": "7"}) == 7.0
    assert parse_retry_after_seconds({}) is None


def test_stratified_selection_spans_full_corpus_without_duplicates():
    idx = select_stratified_indices(total=5000, limit=400)
    assert len(idx) == 400
    assert idx[0] == 0
    assert idx[-1] == 4999
    assert idx == sorted(set(idx))
    assert 2000 < idx[200] < 3000


def test_checkpoint_roundtrip_and_idempotent_key_overwrite(tmp_path):
    cp = JsonlCheckpoint(tmp_path / "checkpoint.jsonl", key_field="chunk_id")
    cp.upsert({"chunk_id": "c1", "value": 1})
    cp.upsert({"chunk_id": "c2", "value": 2})
    cp.upsert({"chunk_id": "c1", "value": 3})

    assert cp.completed_keys() == {"c1", "c2"}
    rows = cp.rows()
    assert rows == [
        {"chunk_id": "c1", "value": 3},
        {"chunk_id": "c2", "value": 2},
    ]


def test_golden_dataset_has_50_complete_questions_and_all_groups():
    df = load_golden_dataset("data/graphrag_golden_50_first5000.csv")
    report = validate_golden_dataset(df)
    assert report["rows"] == 50
    assert report["missing_reference_answers"] == 0
    assert report["missing_reference_evidence"] == 0
    assert set(report["groups"]) == {"factoid", "multi-hop", "cross-doc"}


def test_golden_validation_rejects_blank_reference_answer():
    bad = pd.DataFrame([
        {
            "id": "X1",
            "group": "factoid",
            "question": "q",
            "reference_answer": " ",
            "reference_evidence": "row 1",
        }
    ])
    with pytest.raises(ValueError, match="reference_answer"):
        validate_golden_dataset(bad)
