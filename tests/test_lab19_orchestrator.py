from lab19_orchestrator import effective_run_config


def test_full_config_stays_within_groq_free_tier_budget_strategy():
    config = effective_run_config("full")
    assert config.golden_limit == 50
    assert 100 <= config.extraction_max_chunks <= 160
    assert config.groq_min_interval_s >= 20.0
    assert config.coref_batch_size == 4
    assert config.extraction_batch_size == 4


def test_smoke_config_is_small_but_keeps_same_pipeline():
    config = effective_run_config("smoke")
    assert config.golden_limit == 3
    assert config.extraction_max_chunks == 24
    assert config.groq_min_interval_s >= 20.0
