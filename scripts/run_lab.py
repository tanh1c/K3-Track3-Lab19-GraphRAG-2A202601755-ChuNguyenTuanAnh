from __future__ import annotations

import argparse
import json

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
