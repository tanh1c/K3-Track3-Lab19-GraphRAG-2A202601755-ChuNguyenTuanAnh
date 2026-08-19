from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Callable

from lab19_models import DEFAULT_LLM_PROVIDER, DEFAULT_OPENAI_PIPELINE_MODEL


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def run_check(name: str, fn: Callable[[], dict]) -> tuple[dict, bool]:
    print(f"[preflight] Checking {name}...")
    t0 = time.perf_counter()
    try:
        result = dict(fn())
        result.update({"ok": True, "latency_s": round(time.perf_counter() - t0, 3)})
        print(f"[preflight] {name}: OK")
        return result, True
    except Exception as exc:
        result = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "latency_s": round(time.perf_counter() - t0, 3),
        }
        print(f"[preflight] {name}: FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return result, False


def main() -> int:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    neo4j_database = os.getenv("NEO4J_DATABASE", "").strip()
    llm_provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
    pipeline_model = os.getenv("OPENAI_PIPELINE_MODEL", DEFAULT_OPENAI_PIPELINE_MODEL).strip()
    judge_provider = os.getenv("JUDGE_PROVIDER", "openai").strip().lower()
    judge_model = os.getenv("JUDGE_MODEL", "").strip()
    report: dict[str, object] = {
        "mode": "preflight",
        "checks": {},
        "secrets_printed": False,
        "configuration": {
            "neo4j_database": neo4j_database,
            "llm_provider": llm_provider,
            "pipeline_model": pipeline_model,
            "judge_provider": judge_provider,
            "judge_model": judge_model,
            "source_policy": "FIRST_5000_ROWS_ONLY",
        },
    }

    def check_neo4j() -> dict:
        from neo4j import GraphDatabase

        user = required_env("NEO4J_USER")
        database = required_env("NEO4J_DATABASE")
        driver = GraphDatabase.driver(
            required_env("NEO4J_URI"),
            auth=(user, required_env("NEO4J_PASSWORD")),
        )
        try:
            driver.verify_connectivity()
            with driver.session(database=database) as session:
                value = session.run("RETURN 1 AS ok").single()["ok"]
            if value != 1:
                raise RuntimeError("Neo4j sanity query returned unexpected result")
            return {"database": database, "username_present": bool(user)}
        finally:
            driver.close()

    def check_huggingface() -> dict:
        from datasets import load_dataset

        ds = load_dataset(
            "HackerNoon/tech-company-news-data-dump",
            split="train",
            streaming=True,
            token=required_env("HF_TOKEN"),
        )
        iterator = iter(ds)
        first, second = next(iterator), next(iterator)
        columns = list(first.keys())
        if not columns or set(first.keys()) != set(second.keys()):
            raise RuntimeError("Hugging Face stream returned inconsistent schema")
        return {"columns": columns}

    def check_openai() -> dict:
        if llm_provider != "openai":
            raise RuntimeError(f"Primary LLM provider must be openai, got {llm_provider!r}")
        from openai import OpenAI

        client = OpenAI(api_key=required_env("OPENAI_API_KEY"), timeout=60.0, max_retries=3)
        response = client.chat.completions.create(
            model=pipeline_model,
            messages=[{"role": "user", "content": "Reply with exactly OK."}],
            temperature=0.0,
            max_tokens=8,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("OpenAI returned an empty response")
        usage = getattr(response, "usage", None)
        return {
            "provider": "openai",
            "model": pipeline_model,
            "response_nonempty": True,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        }

    def check_judge_configuration() -> dict:
        if judge_provider != "openai":
            raise RuntimeError("This submission uses OpenAI for the LLM Judge")
        required_env("OPENAI_API_KEY")
        if not judge_model:
            raise RuntimeError("JUDGE_MODEL is empty")
        return {"provider": judge_provider, "model": judge_model, "credential_present": True}

    overall = True
    for key, label, fn in [
        ("neo4j", "Neo4j connectivity", check_neo4j),
        ("huggingface", "Hugging Face streaming access", check_huggingface),
        ("openai", "OpenAI primary LLM", check_openai),
        ("judge", "Judge configuration", check_judge_configuration),
    ]:
        result, ok = run_check(label, fn)
        report["checks"][key] = result
        overall = overall and ok

    report["ok"] = overall
    (out_dir / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[preflight] Sanitized report written to outputs/preflight.json")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
