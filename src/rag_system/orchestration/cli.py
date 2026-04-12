# CLI: router + dispatch + RAG (DOCUMENT_SEARCH) or informational branch.

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from rag_system.retrieval import RetrieverConfig

from .runner import OrchestrationResult, run_orchestrated_query


def _configure_utf8_output() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def _result_to_json_payload(r: OrchestrationResult) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "query": None,
        "intent": r.intent,
        "dispatch": asdict(r.plan),
        "note": r.note,
        "error": r.error,
    }
    if r.trace is not None:
        payload["route_trace"] = asdict(r.trace)
    if r.context is not None:
        payload["context"] = r.context
    if r.answer is not None:
        payload["answer"] = r.answer.answer
        payload["citations"] = [asdict(c) for c in r.answer.citations]
    if r.verification is not None:
        payload["verification"] = {
            "is_grounded": r.verification.is_grounded,
            "is_complete": r.verification.is_complete,
            "issues": [asdict(i) for i in r.verification.issues],
        }
    return payload


def cmd_run(args: argparse.Namespace) -> int:
    db = Path(args.db)
    cfg = RetrieverConfig(mode=args.mode, top_k=args.top_k, alpha=args.alpha)
    result = run_orchestrated_query(
        args.query,
        db,
        retriever=cfg,
        verify=args.verify,
        skip_router=args.no_route,
        llm_router_fallback=args.llm_route,
    )

    if args.json:
        payload = _result_to_json_payload(result)
        payload["query"] = args.query
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if not result.error else 2

    if result.trace is not None and args.debug:
        t = result.trace
        print("[route trace]", file=sys.stderr)
        print(f"  normalized: {t.normalized_query!r}", file=sys.stderr)
        print(f"  scores: {t.scores}", file=sys.stderr)
        print(f"  decision: {t.decision}", file=sys.stderr)

    print(f"intent: {result.intent}")
    print(
        f"dispatch: should_call_api={result.plan.should_call_api} "
        f"api_name={result.plan.api_name!r}",
    )

    if result.note:
        print(f"\n{result.note}")
        return 0

    if result.error:
        print(f"\n[ERROR] {result.error}", file=sys.stderr)
        return 2

    assert result.answer is not None
    print("\n========================================================\n")
    print(f"Question: {args.query}\n")
    print("========================================================\n")
    print("Answer:\n")
    print(result.answer.answer)
    print("\n========================================================\n")
    print("Sources:")
    if not result.answer.citations:
        print("  (no citations detected in answer)")
    else:
        for c in result.answer.citations:
            src = c.doc_source or ""
            page = f", page {c.page}" if c.page is not None else ""
            print(f"  {c.label} chunk_id={c.chunk_id} source={src}{page}")

    if result.verification is not None:
        v = result.verification
        print("\nVerification:")
        print(f"  grounded: {v.is_grounded}")
        print(f"  complete: {v.is_complete}")
        if not v.issues:
            print("  issues: (none)")
        else:
            print("  issues:")
            for i in v.issues:
                labels = ", ".join(i.citation_labels) if i.citation_labels else "none"
                print(f"    - type={i.type}, citations={labels}")
                print(f"      {i.description}")

    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Orchestrated pipeline: intent router → dispatch → RAG or branch message.",
    )
    ap.add_argument("--db", required=True, help="SQLite chunk store path.")
    ap.add_argument("--query", "-q", required=True, help="User question.")
    ap.add_argument("--top-k", type=int, default=6, help="Context chunks for DOCUMENT_SEARCH.")
    ap.add_argument(
        "--mode",
        choices=["semantic", "lexical", "hybrid"],
        default="hybrid",
        help="Retriever mode when intent is DOCUMENT_SEARCH.",
    )
    ap.add_argument("--alpha", type=float, default=0.6, help="Hybrid semantic weight (0–1).")
    ap.add_argument("--json", action="store_true", help="Print structured JSON.")
    ap.add_argument("--verify", action="store_true", help="Run verifier after answer (DOCUMENT_SEARCH).")
    ap.add_argument(
        "--no-route",
        action="store_true",
        help="Skip the router; always run DOCUMENT_SEARCH (same as qa.py).",
    )
    ap.add_argument(
        "--llm-route",
        action="store_true",
        help="OpenAI-compatible LLM fallback when the rule router returns UNKNOWN.",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Print route trace to stderr (scores, normalized query).",
    )
    ap.set_defaults(func=cmd_run)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_utf8_output()
    args = parse_args(argv)
    return args.func(args)
