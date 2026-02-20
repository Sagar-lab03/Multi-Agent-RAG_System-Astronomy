# Purpose: entry point for routing.
# Enables python -m rag_system.routing "query"

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

