# Purpose: SQLite chunk store for ingested documents and chunks.
# Provides: schema init, replace-all write (full refresh), and read helpers for retrieval.

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .models import Chunk, Document


# open DB connection
def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            source_ext TEXT NOT NULL,
            metadata_json TEXT,
            ingested_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
    """)


def replace_all(
    conn: sqlite3.Connection,
    documents: Sequence[Document],
    all_chunks: Sequence[Chunk],
    *,
    ingested_at: Optional[str] = None,
) -> None:
    """Full refresh: delete existing data, then insert all documents and chunks."""
    import datetime as dt
    ts = ingested_at or dt.datetime.now(dt.timezone.utc).isoformat()

    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM documents")

    for doc in documents:
        meta = dict(doc.metadata)
        conn.execute(
            "INSERT INTO documents (doc_id, source_path, source_ext, metadata_json, ingested_at) VALUES (?, ?, ?, ?, ?)",
            (doc.doc_id, doc.source_path, doc.source_ext, json.dumps(meta), ts),
        )
    for c in all_chunks:
        conn.execute(
            "INSERT INTO chunks (chunk_id, doc_id, chunk_index, text, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (c.chunk_id, c.doc_id, c.chunk_index, c.text, json.dumps(c.metadata), ts),
        )
    conn.commit()


def insert_document_and_chunks(
    conn: sqlite3.Connection,
    doc: Document,
    chunks: Sequence[Chunk],
    *,
    ingested_at: Optional[str] = None,
) -> None:
    """Insert one document and its chunks (for incremental or single-doc ingest)."""
    import datetime as dt
    ts = ingested_at or dt.datetime.now(dt.timezone.utc).isoformat()
    meta = dict(doc.metadata)
    conn.execute(
        "INSERT OR REPLACE INTO documents (doc_id, source_path, source_ext, metadata_json, ingested_at) VALUES (?, ?, ?, ?, ?)",
        (doc.doc_id, doc.source_path, doc.source_ext, json.dumps(meta), ts),
    )
    for c in chunks:
        conn.execute(
            "INSERT OR REPLACE INTO chunks (chunk_id, doc_id, chunk_index, text, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (c.chunk_id, c.doc_id, c.chunk_index, c.text, json.dumps(c.metadata), ts),
        )
    conn.commit()


def get_all_chunks(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Return all chunks as list of dicts (chunk_id, doc_id, chunk_index, text, metadata, doc_source, doc_type)."""
    cur = conn.execute("""
        SELECT c.chunk_id, c.doc_id, c.chunk_index, c.text, c.metadata_json, d.source_path, d.source_ext
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        ORDER BY c.doc_id, c.chunk_index
    """)
    rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
        out.append({
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "chunk_index": r["chunk_index"],
            "text": r["text"],
            "metadata": meta,
            "doc_source": r["source_path"],
            "doc_type": (r["source_ext"] or "").lstrip("."),
        })
    return out


def get_chunks_by_doc_id(conn: sqlite3.Connection, doc_id: str) -> List[Dict[str, Any]]:
    """Return chunks for one document."""
    cur = conn.execute("""
        SELECT c.chunk_id, c.doc_id, c.chunk_index, c.text, c.metadata_json, d.source_path, d.source_ext
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.doc_id = ?
        ORDER BY c.chunk_index
    """, (doc_id,))
    rows = cur.fetchall()
    return [
        {
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "chunk_index": r["chunk_index"],
            "text": r["text"],
            "metadata": json.loads(r["metadata_json"]) if r["metadata_json"] else {},
            "doc_source": r["source_path"],
            "doc_type": (r["source_ext"] or "").lstrip("."),
        }
        for r in rows
    ]


def get_document_count(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM documents")
    return cur.fetchone()[0]


def get_chunk_count(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM chunks")
    return cur.fetchone()[0]
