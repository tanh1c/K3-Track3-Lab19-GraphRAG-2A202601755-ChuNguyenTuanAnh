"""Groq request policy for the Lab 19 free-tier workflow.

GPT-OSS is a reasoning model. We request low reasoning effort and exclude the
reasoning field so extraction spends its token budget on the schema-bearing
final answer rather than verbose hidden/returned reasoning.
"""

from __future__ import annotations

from typing import Any


def build_groq_request_options(model: str, *, max_tokens: int) -> dict[str, Any]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    options: dict[str, Any] = {"max_completion_tokens": int(max_tokens)}
    if str(model).startswith("openai/gpt-oss-"):
        options.update(
            {
                "reasoning_effort": "low",
                "include_reasoning": False,
            }
        )
    return options
