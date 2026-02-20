# Purpose: entry point for ingestion.
# Enables python -m rag_system.ingest

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

