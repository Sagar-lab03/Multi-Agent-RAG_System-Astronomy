# CLI: index (embed all chunks) and search (query → top-k).

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .embedding import get_embedder
from .retriever import RetrieverConfig, retrieve_context
from .search import hybrid_search, lexical_search, semantic_search


def _configure_utf8_output() -> None:
    """
    On Windows, the default console encoding can fail on PDF-extracted unicode
    (e.g. ligatures like 'ﬁ'). Configure stdout/stderr to be UTF-8 and replace
    unencodable characters to avoid crashing during printing.
    """
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


def cmd_index(args: argparse.Namespace) -> int:
    from rag_system.ingest.store import get_all_chunks, update_chunk_embeddings_batch

    conn = _get_conn(Path(args.db))
    chunks = get_all_chunks(conn, include_embedding=False)
    if not chunks:
        print("No chunks in DB. Run ingest first with --db.", file=sys.stderr)
        conn.close()
        return 1

    # embedder = HuggingFaceEmbedder()
    embedder = get_embedder()
    batch_size = getattr(args, "batch_size", 32)
    total = len(chunks)
    updated = 0
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        try:
            vectors = embedder.embed_many(texts, batch_size=batch_size)
        except Exception as e:
            print(f"[ERROR] Embedding failed: {e}", file=sys.stderr)
            conn.close()
            return 2
        items = [(c["chunk_id"], vec) for c, vec in zip(batch, vectors)]
        update_chunk_embeddings_batch(conn, items)
        updated += len(items)
        print(f"Embedded {updated}/{total} chunks...", file=sys.stderr)
    conn.close()
    print(f"Indexed {updated} chunks.", file=sys.stderr)
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    conn = _get_conn(Path(args.db))
    try:
        mode = args.mode
        if mode == "semantic":
            embedder = get_embedder()
            results = semantic_search(conn, args.query, embedder, top_k=args.top_k)
        elif mode == "lexical":
            results = lexical_search(conn, args.query, top_k=args.top_k)
        elif mode == "hybrid":
            embedder = get_embedder()
            results = hybrid_search(
                conn, args.query, embedder, top_k=args.top_k, alpha=args.alpha
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        conn.close()
        return 2
    conn.close()
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (mode={args.mode}, score={r.get('score', 0):.4f}) ---")
            print(r.get("text", "")[:500])
            if len(r.get("text", "")) > 500:
                print("...")
            print()
    return 0


def cmd_repl(args: argparse.Namespace) -> int:
    """
    Simple interactive loop to try different queries in a single process.
    Reuses the same DB connection and BM25 cache.
    """
    conn = _get_conn(Path(args.db))
    mode = args.mode
    alpha = args.alpha
    top_k = args.top_k

    embedder = None
    if mode in ("semantic", "hybrid"):
        embedder = get_embedder()

    print(f"Retrieval REPL (mode={mode}, top_k={top_k}, alpha={alpha})")
    print("Type a query and press Enter. Type 'exit' or 'quit' to leave.")

    try:
        while True:
            try:
                line = input("query> ")
            except EOFError:
                break
            q = (line or "").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit", "q"}:
                break

            try:
                if mode == "semantic":
                    if embedder is None:
                        embedder = get_embedder()
                    results = semantic_search(conn, q, embedder, top_k=top_k)
                elif mode == "lexical":
                    results = lexical_search(conn, q, top_k=top_k)
                elif mode == "hybrid":
                    if embedder is None:
                        embedder = get_embedder()
                    results = hybrid_search(conn, q, embedder, top_k=top_k, alpha=alpha)
                else:
                    print(f"Unknown mode: {mode}", file=sys.stderr)
                    continue
            except Exception as e:
                print(f"[ERROR] {e}", file=sys.stderr)
                continue

            for i, r in enumerate(results, 1):
                print(f"\n--- Result {i} (mode={mode}, score={r.get('score', 0):.4f}) ---")
                print(r.get("text", "")[:500])
                if len(r.get("text", "")) > 500:
                    print("...")
                print()
    finally:
        conn.close()
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    """
    One-shot retriever call: show the chunks that the retriever agent would use
    for a given query (mode + top_k + alpha).
    """
    conn = _get_conn(Path(args.db))
    cfg = RetrieverConfig(mode=args.mode, top_k=args.top_k, alpha=args.alpha)
    try:
        results = retrieve_context(conn, args.query, cfg)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        conn.close()
        return 2
    conn.close()

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {}) or {}
        page = meta.get("page")
        src = r.get("doc_source", "")
        score = r.get("score", 0.0)
        header = f"\n--- Context {i} (mode={args.mode}, score={score:.4f}) ---"
        if src:
            header += f"\nsource: {src}"
        if page is not None:
            header += f" (page {page})"
        print(header)
        print(r.get("text", "")[:800])
        if len(r.get("text", "")) > 800:
            print("...")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Retrieval: index embeddings and search.")
    sub = ap.add_subparsers(dest="command", required=True)
    idx = sub.add_parser("index", help="Embed all chunks and store in DB.")
    idx.add_argument("--db", required=True, help="SQLite chunk store path.")
    idx.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding API.")
    idx.set_defaults(func=cmd_index)
    sch = sub.add_parser("search", help="Run a search query (semantic, lexical, or hybrid).")
    sch.add_argument("--db", required=True, help="SQLite chunk store path.")
    sch.add_argument("--query", "-q", required=True, help="Search query.")
    sch.add_argument("--top-k", type=int, default=5, help="Number of results.")
    sch.add_argument(
        "--mode",
        choices=["semantic", "lexical", "hybrid"],
        default="hybrid",
        help="Search mode. Default: hybrid.",
    )
    sch.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Hybrid weight for semantic score (0-1). Used only in hybrid mode.",
    )
    sch.add_argument("--json", action="store_true", help="Output full JSON.")
    sch.set_defaults(func=cmd_search)

    repl = sub.add_parser("repl", help="Interactive search REPL (shares cache/index).")
    repl.add_argument("--db", required=True, help="SQLite chunk store path.")
    repl.add_argument("--top-k", type=int, default=5, help="Number of results.")
    repl.add_argument(
        "--mode",
        choices=["semantic", "lexical", "hybrid"],
        default="hybrid",
        help="Search mode. Default: hybrid.",
    )
    repl.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Hybrid weight for semantic score (0-1). Used only in hybrid mode.",
    )
    repl.set_defaults(func=cmd_repl)

    ctx = sub.add_parser(
        "context", help="Show retriever-selected context chunks for a query."
    )
    ctx.add_argument("--db", required=True, help="SQLite chunk store path.")
    ctx.add_argument("--query", "-q", required=True, help="Query to retrieve for.")
    ctx.add_argument("--top-k", type=int, default=6, help="Number of chunks to retrieve.")
    ctx.add_argument(
        "--mode",
        choices=["semantic", "lexical", "hybrid"],
        default="hybrid",
        help="Retriever mode. Default: hybrid.",
    )
    ctx.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Hybrid weight for semantic score (0-1). Used only in hybrid mode.",
    )
    ctx.add_argument("--json", action="store_true", help="Output full JSON.")
    ctx.set_defaults(func=cmd_context)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_utf8_output()
    args = parse_args(argv)
    return args.func(args)
