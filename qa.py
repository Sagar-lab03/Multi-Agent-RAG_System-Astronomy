# Entry point for the question-answering CLI tool in the RAG system.
# This script ensures the src directory is on the Python path and then calls the CLI main function 
#      defined in rag_system.qa.cli.

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
    from rag_system.qa.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())

