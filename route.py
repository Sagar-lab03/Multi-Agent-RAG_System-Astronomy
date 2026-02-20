# Purpose: entry point for routing. Same convenience pattern as ingest.py, but for routing.
# Enables python -m rag_system.routing "query".
# Adds src/ to sys.path and runs rag_system.routing.cli.

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _ensure_src_on_path()
    from rag_system.routing.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())

