"""Run-mode policy for Lab 19.

This module is intentionally tiny: it applies provider-budget guardrails on top
of the shared runtime config so smoke/full execution uses the same pipeline but
cannot accidentally hammer the Groq free tier.
"""

from __future__ import annotations

from dataclasses import replace

from lab19_runtime import RunConfig


def effective_run_config(mode: str) -> RunConfig:
    """Return a rate-limit-safe config for the requested execution mode.

    The source corpus remains the first 5,000 rows in every mode. Smoke only
    reduces expensive LLM/evaluation work. Full mode caps graph extraction at
    160 distributed chunks and paces Groq at no more than roughly 3 requests
    per minute before provider-driven Retry-After/backoff is applied.
    """
    config = RunConfig.for_mode(mode)
    common = {
        "groq_timeout_s": max(config.groq_timeout_s, 60.0),
        "groq_min_interval_s": max(config.groq_min_interval_s, 20.0),
        "groq_max_retries": max(config.groq_max_retries, 7),
        "coref_batch_size": 4,
        "extraction_batch_size": 4,
    }
    if config.mode == "full":
        return replace(config, extraction_max_chunks=160, **common)
    return replace(config, **common)
