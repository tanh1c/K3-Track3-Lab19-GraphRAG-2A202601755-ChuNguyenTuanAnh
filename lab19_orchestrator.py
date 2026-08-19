"""Run-mode policy for Lab 19 with OpenAI as the primary LLM provider."""

from __future__ import annotations

from dataclasses import replace

from lab19_runtime import RunConfig


def effective_run_config(mode: str) -> RunConfig:
    """Keep assignment scale while removing Groq-specific fixed sleeps.

    Legacy ``groq_*`` field names are retained only for notebook compatibility;
    they now configure the OpenAI-backed runtime. Calls stay sequential and the
    OpenAI SDK handles transient rate limits/retries.
    """
    config = RunConfig.for_mode(mode)
    common = {
        "groq_timeout_s": max(config.groq_timeout_s, 60.0),
        "groq_min_interval_s": 0.0,
        "groq_max_retries": 5,
        "coref_batch_size": 4,
        "extraction_batch_size": 4,
    }
    if config.mode == "full":
        return replace(config, extraction_max_chunks=400, **common)
    return replace(config, **common)
