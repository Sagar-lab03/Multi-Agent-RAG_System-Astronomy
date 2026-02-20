# Purpose: command-line interface for ingestion.
# Contains:
# - iter_files(): recursively find all files in input dir with allowed extensions
# - parse_args(): parse command-line arguments
# - main(): the main entry point

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

from .chunking import chunk_document
from .loaders import load_document
from .models import Chunk, Document
from .store import get_connection, init_schema, replace_all


DEFAULT_EXTS = [".pdf", ".txt", ".md", ".html", ".htm"]


def iter_files(root: Path, *, exts: Set[str]) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts:
            yield p


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Ingest documents; write to SQLite chunk store and/or JSONL."
    )
    ap.add_argument("--input", required=True, help="Input directory containing raw docs.")
    ap.add_argument(
        "--output",
        default=None,
        help="Output JSONL file path (optional; use with --db to write both).",
    )
    ap.add_argument(
        "--db",
        default=None,
        help="SQLite database path for chunk store (e.g. data/processed/chunks.db).",
    )
    ap.add_argument(
        "--exts",
        default=",".join(DEFAULT_EXTS),
        help=f"Comma-separated allowed extensions. Default: {','.join(DEFAULT_EXTS)}",
    )
    ap.add_argument(
        "--encoding",
        default="utf-8",
        help="Text/HTML file encoding (errors are replaced). Default: utf-8",
    )
    ap.add_argument("--max-chars", type=int, default=1200, help="Max chars per chunk.")
    ap.add_argument("--overlap-chars", type=int, default=150, help="Overlap chars.")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not args.output and not args.db:
        print("Provide at least one of --output (JSONL) or --db (SQLite).", file=sys.stderr)
        return 2

    input_dir = Path(args.input)
    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input dir not found: {input_dir}", file=sys.stderr)
        return 2

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.db:
        db_path = Path(args.db)
        db_path.parent.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_chunks = 0
    documents: List[Document] = []
    all_chunks: List[Chunk] = []

    out_file = None
    if args.output:
        out_file = Path(args.output).open("w", encoding="utf-8")

    try:
        for path in iter_files(input_dir, exts=exts):
            total_files += 1
            try:
                doc = load_document(path, encoding=args.encoding)
                chunks = chunk_document(
                    doc, max_chars=args.max_chars, overlap_chars=args.overlap_chars
                )
                if args.db:
                    documents.append(doc)
                    all_chunks.extend(chunks)
                if out_file:
                    for c in chunks:
                        record = {
                            "doc_id": c.doc_id,
                            "doc_source": doc.source_path,
                            "doc_type": doc.source_ext.lstrip("."),
                            "chunk_id": c.chunk_id,
                            "chunk_index": c.chunk_index,
                            "text": c.text,
                            "metadata": c.metadata,
                        }
                        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_chunks += 1
                else:
                    total_chunks += len(chunks)
            except Exception as e:
                print(f"[WARN] Failed to ingest {path}: {e}", file=sys.stderr)
                continue

        if args.db:
            conn = get_connection(Path(args.db))
            try:
                init_schema(conn)
                replace_all(conn, documents, all_chunks)
            finally:
                conn.close()
    finally:
        if out_file:
            out_file.close()

    print(f"Ingested files: {total_files}")
    print(f"Total chunks: {total_chunks}")
    if args.output:
        print(f"JSONL: {args.output}")
    if args.db:
        print(f"DB: {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

