from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> int:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "mode": "preflight",
        "checks": {},
        "secrets_printed": False,
    }

    try:
        neo4j_uri = required_env("NEO4J_URI")
        neo4j_user = required_env("NEO4J_USER")
        neo4j_password = required_env("NEO4J_PASSWORD")
        neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j").strip() or "neo4j"
        groq_key = required_env("GROQ_API_KEY")
        hf_token = required_env("HF_TOKEN")
        groq_fast_model = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant").strip()
        judge_provider = os.getenv("JUDGE_PROVIDER", "openai").strip().lower()
        judge_model = os.getenv("JUDGE_MODEL", "").strip()
        if judge_provider == "openai":
            required_env("OPENAI_API_KEY")

        report["configuration"] = {
            "neo4j_database": neo4j_database,
            "groq_fast_model": groq_fast_model,
            "judge_provider": judge_provider,
            "judge_model": judge_model,
            "source_policy": "FIRST_5000_ROWS_ONLY",
        }

        print("[preflight] Checking Neo4j connectivity...")
        from neo4j import GraphDatabase

        t0 = time.perf_counter()
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        driver.verify_connectivity()
        with driver.session(database=neo4j_database) as session:
            value = session.run("RETURN 1 AS ok").single()["ok"]
        driver.close()
        if value != 1:
            raise RuntimeError("Neo4j sanity query returned unexpected result")
        report["checks"]["neo4j"] = {
            "ok": True,
            "latency_s": round(time.perf_counter() - t0, 3),
        }
        print("[preflight] Neo4j: OK")

        print("[preflight] Checking Hugging Face streaming access...")
        from datasets import load_dataset

        t0 = time.perf_counter()
        ds = load_dataset(
            "HackerNoon/tech-company-news-data-dump",
            split="train",
            streaming=True,
            token=hf_token,
        )
        iterator = iter(ds)
        first = next(iterator)
        second = next(iterator)
        columns = list(first.keys())
        if not columns or set(first.keys()) != set(second.keys()):
            raise RuntimeError("Hugging Face stream returned inconsistent schema")
        report["checks"]["huggingface"] = {
            "ok": True,
            "columns": columns,
            "latency_s": round(time.perf_counter() - t0, 3),
        }
        print(f"[preflight] Hugging Face: OK ({len(columns)} columns)")

        print("[preflight] Checking Groq with a minimal request...")
        from groq import Groq

        t0 = time.perf_counter()
        client = Groq(api_key=groq_key, timeout=30.0, max_retries=0)
        response = client.chat.completions.create(
            model=groq_fast_model,
            messages=[
                {"role": "system", "content": "Return exactly OK."},
                {"role": "user", "content": "Health check"},
            ],
            temperature=0.0,
            max_tokens=4,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("Groq returned an empty response")
        usage = getattr(response, "usage", None)
        report["checks"]["groq"] = {
            "ok": True,
            "model": groq_fast_model,
            "latency_s": round(time.perf_counter() - t0, 3),
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        }
        print("[preflight] Groq: OK")

        report["ok"] = True
        return 0
    except Exception as exc:
        report["ok"] = False
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        print(f"[preflight] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        (out_dir / "preflight.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("[preflight] Sanitized report written to outputs/preflight.json")


if __name__ == "__main__":
    raise SystemExit(main())
