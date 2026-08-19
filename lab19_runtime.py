"""Runtime guardrails for Lab 19.

The full implementation lives in :mod:`lab19_runtime_impl`. This wrapper keeps
that implementation inspectable while applying provider-safe production policy
in one place: current Groq defaults, conservative pacing, resilient JSON
fallback, stable audit CSV schemas, and OpenAI answer synthesis to preserve the
Groq free-tier budget for rubric-critical extraction.
"""

from __future__ import annotations

from dataclasses import replace
import os
import time
from typing import Any, Mapping

import pandas as pd

import lab19_runtime_impl as _impl
from lab19_models import DEFAULT_GROQ_MODEL
from lab19_utils import parse_retry_after_seconds

os.environ.setdefault("GROQ_FAST_MODEL", DEFAULT_GROQ_MODEL)
os.environ.setdefault("GROQ_GENERATION_MODEL", DEFAULT_GROQ_MODEL)

ENTITY_AUDIT_COLUMNS = ["type", "left", "right", "similarity", "decision"]


def ensure_entity_audit_schema(audit: pd.DataFrame) -> pd.DataFrame:
    """Return an audit frame that remains readable even when there are zero rows."""
    if audit is None or audit.empty:
        return pd.DataFrame(columns=ENTITY_AUDIT_COLUMNS)
    out = audit.copy()
    for column in ENTITY_AUDIT_COLUMNS:
        if column not in out.columns:
            out[column] = None
    ordered = ENTITY_AUDIT_COLUMNS + [c for c in out.columns if c not in ENTITY_AUDIT_COLUMNS]
    return out[ordered]


def valid_items_payload(obj: Any) -> bool:
    """Validate the shared coref/extraction envelope before downstream `.get()` calls."""
    if not isinstance(obj, dict):
        return False
    if "items" not in obj:
        return True
    items = obj.get("items")
    return isinstance(items, list) and all(isinstance(item, dict) for item in items)


_original_for_mode = _impl.RunConfig.for_mode


def _safe_for_mode(cls, mode: str):
    base = _original_for_mode(mode)
    common = {
        "groq_timeout_s": max(float(base.groq_timeout_s), 60.0),
        "groq_min_interval_s": max(float(base.groq_min_interval_s), 20.0),
        "groq_max_retries": max(int(base.groq_max_retries), 7),
        "coref_batch_size": 4,
        "extraction_batch_size": 4,
    }
    # Preserve the assignment's 400-chunk full scale guard. The selector spans
    # the complete first-5000 corpus rather than taking a head slice. Evaluation
    # answer generation is routed to OpenAI, preserving Groq daily token budget.
    if str(base.mode).lower() == "full":
        return replace(base, extraction_max_chunks=400, **common)
    return replace(base, **common)


_impl.RunConfig.for_mode = classmethod(_safe_for_mode)


_original_build_resolution_map = _impl.build_resolution_map


def _guarded_build_resolution_map(*args: Any, **kwargs: Any):
    mapping, audit = _original_build_resolution_map(*args, **kwargs)
    return mapping, ensure_entity_audit_schema(audit)


_impl.build_resolution_map = _guarded_build_resolution_map


def _headers_from_exception(exc: Exception) -> Mapping[str, Any] | None:
    response = getattr(exc, "response", None)
    return getattr(response, "headers", None) if response is not None else None


def _safe_groq_chat(
    self,
    *,
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 1200,
    json_mode: bool = False,
):
    """Groq chat with GPT-OSS reasoning hidden and provider-aware backoff."""
    last: Exception | None = None
    chosen_model = model or self.fast_model
    for attempt in range(self.config.groq_max_retries):
        self._pace()
        try:
            kwargs: dict[str, Any] = {
                "model": chosen_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
                "max_completion_tokens": int(max_tokens),
            }
            if chosen_model.startswith("openai/gpt-oss"):
                kwargs["reasoning_effort"] = "low"
                kwargs["reasoning_format"] = "hidden"
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(**kwargs)
            usage_obj = getattr(response, "usage", None)
            usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                "total_tokens": getattr(usage_obj, "total_tokens", None),
            }
            return (response.choices[0].message.content or "").strip(), usage
        except Exception as exc:
            last = exc
            if not self._is_retryable(exc) or attempt == self.config.groq_max_retries - 1:
                raise
            retry_after = parse_retry_after_seconds(_headers_from_exception(exc))
            delay = self.policy.delay_for(attempt, retry_after_s=retry_after)
            print(
                f"[groq] retryable {type(exc).__name__}; "
                f"attempt={attempt + 1}/{self.config.groq_max_retries}; sleep={delay:.1f}s"
            )
            time.sleep(delay)
    raise RuntimeError(last or "Groq retry loop exhausted")


_impl.GroqRuntime.chat = _safe_groq_chat


def _openai_json_fallback(self, *, system: str, user: str, max_tokens: int = 1200, **_: Any):
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Groq JSON output was invalid and OPENAI_API_KEY fallback is unavailable")
    from openai import OpenAI

    model = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    print(f"[llm] Groq JSON validation failed; repairing with OpenAI fallback model={model}")
    client = OpenAI(api_key=key, timeout=60.0, max_retries=3)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
        max_tokens=int(max_tokens),
    )
    text = (response.choices[0].message.content or "").strip()
    obj = _impl.GroqRuntime.parse_json(text)
    if not valid_items_payload(obj):
        raise ValueError("OpenAI JSON fallback returned malformed items envelope")
    usage_obj = getattr(response, "usage", None)
    usage = {
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
        "completion_tokens": getattr(usage_obj, "completion_tokens", None),
        "total_tokens": getattr(usage_obj, "total_tokens", None),
    }
    return obj, usage


def _resilient_chat_json(self, **kwargs: Any):
    try:
        text, usage = self.chat(json_mode=True, **kwargs)
        obj = self.parse_json(text)
        if not valid_items_payload(obj):
            raise ValueError("Malformed items envelope")
        return obj, usage
    except Exception as exc:
        message = str(exc).lower()
        repairable = (
            "json_validate_failed" in message
            or "failed to generate json" in message
            or "failed to validate json" in message
            or "malformed items envelope" in message
            or "no json object" in message
        )
        if not repairable:
            raise
        return _openai_json_fallback(self, **kwargs)


_impl.GroqRuntime.chat_json = _resilient_chat_json


_original_generate_answer = _impl.generate_answer


def _openai_generate_answer(llm, question: str, context: str):
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return _original_generate_answer(llm, question, context)
    from openai import OpenAI

    model = os.getenv("OPENAI_GENERATION_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    client = OpenAI(api_key=key, timeout=60.0, max_retries=3)
    system = (
        "Answer only from supplied context. Be concise but complete. Do not invent facts. "
        "Cite provenance inline as [chunk_id=...] or [source_row=...] whenever possible. "
        "If evidence is insufficient or conflicting, state that explicitly."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:"},
        ],
        temperature=0.0,
        max_tokens=900,
    )
    usage_obj = getattr(response, "usage", None)
    usage = {
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
        "completion_tokens": getattr(usage_obj, "completion_tokens", None),
        "total_tokens": getattr(usage_obj, "total_tokens", None),
    }
    return (response.choices[0].message.content or "").strip(), usage


_impl.generate_answer = _openai_generate_answer

from lab19_runtime_impl import *  # noqa: E402,F401,F403
