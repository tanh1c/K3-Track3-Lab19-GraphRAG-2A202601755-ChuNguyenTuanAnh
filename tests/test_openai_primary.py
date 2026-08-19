from lab19_models import DEFAULT_LLM_PROVIDER, DEFAULT_OPENAI_PIPELINE_MODEL
from lab19_runtime import OpenAIRuntime, RunConfig


def test_openai_is_primary_pipeline_provider():
    assert DEFAULT_LLM_PROVIDER == "openai"
    assert DEFAULT_OPENAI_PIPELINE_MODEL == "gpt-4.1-mini"
    assert OpenAIRuntime.__name__ == "OpenAIRuntime"


def test_openai_primary_runtime_has_no_artificial_groq_sleep():
    smoke = RunConfig.for_mode("smoke")
    full = RunConfig.for_mode("full")
    assert smoke.groq_min_interval_s <= 1.0
    assert full.groq_min_interval_s <= 1.0
    assert full.extraction_max_chunks == 400
    assert full.golden_limit == 50
