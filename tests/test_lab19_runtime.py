import pandas as pd

from lab19_runtime import DEFAULT_GROQ_MODEL, GroqRuntime, RunConfig, comparison_table


def test_default_groq_model_uses_current_post_deprecation_replacement():
    assert DEFAULT_GROQ_MODEL == "openai/gpt-oss-20b"


def test_run_modes_keep_full_scope_but_reduce_only_expensive_smoke_work():
    smoke = RunConfig.for_mode("smoke")
    full = RunConfig.for_mode("full")
    assert smoke.extraction_max_chunks == 24
    assert smoke.golden_limit == 3
    assert full.extraction_max_chunks == 400
    assert full.golden_limit == 50
    assert smoke.groq_timeout_s == 45.0
    assert smoke.groq_min_interval_s >= 10.0


def test_parse_json_accepts_fenced_json():
    parsed = GroqRuntime.parse_json('```json\n{"ok": true}\n```')
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
