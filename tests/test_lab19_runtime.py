import pandas as pd

from lab19_groq import build_groq_request_options
from lab19_models import DEFAULT_GROQ_MODEL, DEFAULT_LLM_PROVIDER, DEFAULT_OPENAI_PIPELINE_MODEL
from lab19_runtime import OpenAIRuntime, RunConfig, comparison_table


def test_openai_is_primary_pipeline_provider():
    assert DEFAULT_LLM_PROVIDER == "openai"
    assert DEFAULT_OPENAI_PIPELINE_MODEL == "gpt-4.1-mini"
    assert OpenAIRuntime.__name__ == "OpenAIRuntime"


def test_optional_groq_default_uses_current_post_deprecation_replacement():
    assert DEFAULT_GROQ_MODEL == "openai/gpt-oss-20b"


def test_gpt_oss_helper_remains_available_for_optional_fallback():
    options = build_groq_request_options("openai/gpt-oss-20b", max_tokens=64)
    assert options["reasoning_effort"] == "low"
    assert options["include_reasoning"] is False
    assert options["max_completion_tokens"] == 64
    assert "max_tokens" not in options


def test_run_modes_keep_full_scope_without_artificial_groq_sleep():
    smoke = RunConfig.for_mode("smoke")
    full = RunConfig.for_mode("full")
    assert smoke.extraction_max_chunks == 24
    assert smoke.golden_limit == 3
    assert full.extraction_max_chunks == 400
    assert full.golden_limit == 50
    assert smoke.groq_timeout_s >= 60.0
    assert smoke.groq_min_interval_s <= 1.0
    assert full.groq_timeout_s >= 60.0
    assert full.groq_min_interval_s <= 1.0


def test_parse_json_accepts_fenced_json():
    parsed = OpenAIRuntime.parse_json('```json\n{"ok": true}\n```')
    assert parsed == {"ok": True}


def test_comparison_table_exports_all_required_metrics_by_group():
    eval_df = pd.DataFrame(
        [
            {
                "group": "multi-hop",
                "flat_comprehensiveness": 2,
                "graph_comprehensiveness": 4,
                "flat_faithfulness": 4,
                "graph_faithfulness": 5,
                "flat_multi_hop_reasoning": 2,
                "graph_multi_hop_reasoning": 5,
                "flat_latency_s": 1.0,
                "graph_latency_s": 2.0,
                "flat_total_tokens": 100,
                "graph_total_tokens": 180,
            }
        ]
    )
    out = comparison_table(eval_df)
    assert set(out.metric) == {
        "Comprehensiveness",
        "Faithfulness",
        "Multi-hop reasoning",
        "Latency (s)",
        "Token usage",
    }
    comp = out[out.metric.eq("Comprehensiveness")].iloc[0]
    assert comp.delta_graph_minus_flat == 2.0
