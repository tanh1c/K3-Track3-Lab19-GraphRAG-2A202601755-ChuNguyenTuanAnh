from __future__ import annotations

import argparse
import json
import os

from lab19_models import DEFAULT_GROQ_MODEL

# Ensure local/Colab execution cannot silently fall back to the Llama model IDs
# retired by Groq on 2026-08-16.
os.environ.setdefault("GROQ_FAST_MODEL", DEFAULT_GROQ_MODEL)
os.environ.setdefault("GROQ_GENERATION_MODEL", DEFAULT_GROQ_MODEL)

from lab19_runtime import run_lab


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Lab 19 GraphRAG pipeline")
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    manifest = run_lab(args.mode)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
