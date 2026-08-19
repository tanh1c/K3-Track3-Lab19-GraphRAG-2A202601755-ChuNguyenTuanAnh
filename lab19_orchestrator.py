"""Run-mode policy for Lab 19.

This adapter mirrors the runtime guardrails for callers that want an explicit
configuration object without executing the pipeline.
"""

from __future__ import annotations

from dataclasses import replace

from lab19_runtime import RunConfig


def effective_run_config(mode: str) -> RunConfig:
    """Return the assignment-scale configuration with conservative Groq pacing."""
    config = RunConfig.for_mode(mode)
    common = {
        "groq_timeout_s": max(config.groq_timeout_s, 60.0),
        "groq_min_interval_s": max(config.groq_min_interval_s, 20.0),
        "groq_max_retries": max(config.groq_max_retries, 7),
        "coref_batch_size": 4,
        "extraction_batch_size": 4,
    }
    if config.mode == "full":
        return replace(config, extraction_max_chunks=400, **common)
    return replace(config, **common)
