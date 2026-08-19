import pandas as pd
import pytest

from lab19_utils import (
    FIRST_N_ROWS,
    RateLimitPolicy,
    load_golden_dataset,
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
