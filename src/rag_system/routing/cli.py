# Purpose: command-line interface for routing.
# Guarantee: stdout prints the label only; debug trace goes to stderr.
# Flags:
# --debug: JSON trace to stderr
# --enable-llm-fallback: allow LLM fallback when rule-based returns UNKNOWN

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .router import ConstrainedRouter


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Route a query into exactly one intent label."
    )
    ap.add_argument("query", help="User query to route (string).")
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Write a JSON debug trace to stderr (stdout stays label-only).",
    )
    ap.add_argument(
        "--enable-llm-fallback",
        action="store_true",
        help="Enable optional prompt-based LLM fallback when rule router returns UNKNOWN.",
    )
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    router = ConstrainedRouter(enable_llm_fallback=args.enable_llm_fallback)
    label, trace = router.route_with_trace(args.query)

    if args.debug:
        # Debug goes to stderr so stdout can remain label-only.
        print(json.dumps(trace.__dict__, ensure_ascii=False), file=sys.stderr)

    # CRITICAL: label-only output to stdout.
    print(label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

