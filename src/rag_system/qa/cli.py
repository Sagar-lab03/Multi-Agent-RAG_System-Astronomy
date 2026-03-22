# Command-line interface for question-answering pipeline (retriever + Gemini answerer).
# Provides a CLI for running the QA pipeline with customizable parameters.
# Uses argparse for command-line argument parsing and JSON output.
# Example usage:
#   python cli.py --db data/processed/chunks.db --query "What is the distance to the nearest star?"
#   python cli.py --db data/processed/chunks.db --query "What is the distance to the nearest star?" --json
#   python cli.py --db data/processed/chunks.db --query "What is the distance to the nearest star?" --top-k 10 --mode semantic --alpha 0.8
#   python cli.py --db data/processed/chunks.db --query "What is the distance to the nearest star?" --top-k 10 --mode semantic --alpha 0.8 --json
#   python cli.py --db data/processed/chunks.db --query "What is the distance to the nearest star?" --top-k 10 --mode semantic --alpha 0.8 --json --mode lexical
#   python cli.py --db data/processed/chunks.db --query "What is the distance to the nearest star?" --top-k 10 --mode semantic --alpha 0.8 --json --mode hybrid

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from rag_system.retrieval import RetrieverConfig, retrieve_context

from .answerer import AnswerConfig, AnswerWithCitations, answer_question
from .verifier import VerificationResult, verify_answer


def _configure_utf8_output() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def _get_conn(db_path: Path):
    from rag_system.ingest.store import get_connection, init_schema

    conn = get_connection(db_path)
    init_schema(conn)
    return conn


def cmd_answer(args: argparse.Namespace) -> int:
    conn = _get_conn(Path(args.db))
    cfg = RetrieverConfig(mode=args.mode, top_k=args.top_k, alpha=args.alpha)
    try:
        context = retrieve_context(conn, args.query, cfg)
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}", file=sys.stderr)
        conn.close()
        return 2
    conn.close()

    if not context:
        print("[WARN] No context chunks retrieved; cannot answer reliably.", file=sys.stderr)
        return 1

    try:
        acfg = AnswerConfig()
        result: AnswerWithCitations = answer_question(args.query, context, config=acfg)
    except Exception as e:
        print(f"[ERROR] Answer generation failed: {e}", file=sys.stderr)
        return 2
    ver: Optional[VerificationResult] = None
    if args.verify:
        try:
            ver = verify_answer(args.query, result.answer, context)
        except Exception as e:
            print(f"[WARN] Verification failed: {e}", file=sys.stderr)
            ver = None

    if args.json:
        payload = {
            "query": args.query,
            "answer": result.answer,
            "citations": [c.__dict__ for c in result.citations],
            "context": context,
        }
        if ver is not None:
            payload["verification"] = {
                "is_grounded": ver.is_grounded,
                "is_complete": ver.is_complete,
                "issues": [i.__dict__ for i in ver.issues],
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("========================================================\n")
    print(f"Question: {args.query}\n")
    print("========================================================\n")
    print("Answer:\n")
    print(result.answer)
    print("========================================================\n")
    print("Sources:")
    if not result.citations:
        print("  (no citations detected in answer)")
    else:
        for c in result.citations:
            src = c.doc_source or ""
            page = f", page {c.page}" if c.page is not None else ""
            print(f"  {c.label} chunk_id={c.chunk_id} source={src}{page}")

    if ver is not None:
        print("\nVerification:")
        print(f"  grounded: {ver.is_grounded}")
        print(f"  complete: {ver.is_complete}")
        if not ver.issues:
            print("  issues: (none)")
        else:
            print("  issues:")
            for i in ver.issues:
                labels = ", ".join(i.citation_labels) if i.citation_labels else "none"
                print(f"    - type={i.type}, citations={labels}")
                print(f"      {i.description}")

    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Question answering pipeline (retriever + Gemini answerer).")
    ap.add_argument("--db", required=True, help="SQLite chunk store path.")
    ap.add_argument("--query", "-q", required=True, help="User question.")
    ap.add_argument("--top-k", type=int, default=6, help="Number of context chunks to retrieve.")
    ap.add_argument(
        "--mode",
        choices=["semantic", "lexical", "hybrid"],
        default="hybrid",
        help="Retriever mode. Default: hybrid.",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Hybrid weight for semantic score (0-1). Used only in hybrid mode.",
    )
    ap.add_argument("--json", action="store_true", help="Output full JSON.")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Run a verification pass (groundedness/completeness) after answering.",
    )
    ap.set_defaults(func=cmd_answer)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_utf8_output()
    args = parse_args(argv)
    return args.func(args)

