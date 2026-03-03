# Purpose: SQLite chunk store for ingested documents and chunks.
# Provides: schema init, replace-all write (full refresh), and read helpers for retrieval.

from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .models import Chunk, Document


''' --------------------------------------------------
Convert list[float] embedding -> compact float32 binary (BLOB).
Each float uses 4 bytes (float32).
`*embedding` expands the list into positional args for struct.pack().

Before:
  embedding = [1.0, 2.0]

After:
  blob = struct.pack("2f", *embedding)
  # blob -> b'...'  (8 bytes total: 2 floats × 4 bytes)
------------------------------------------------------- ''' 
def _embedding_to_blob(embedding: List[float]) -> bytes:
    """Pack float list as little-endian float32 BLOB."""
    return struct.pack(f"{len(embedding)}f", *embedding)


''' --------------------------------------------------
Convert float32 binary (BLOB) -> list[float].
Number of floats = len(blob) // 4 (since each float32 = 4 bytes).

Before:
  blob = struct.pack("2f", 1.0, 2.0)

After:
  embedding = list(struct.unpack("2f", blob))
  # embedding -> [1.0, 2.0]
------------------------------------------------------- ''' 
def _blob_to_embedding(blob: Optional[bytes]) -> Optional[List[float]]:
    """Unpack BLOB to list of floats."""
    if blob is None:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


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
    ensure_embedding_column(conn)


def ensure_embedding_column(conn: sqlite3.Connection) -> None:
    """Add embedding BLOB column to chunks if missing (safe for existing DBs)."""
    try:
        conn.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB")
        conn.commit()
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            pass
        else:
            raise


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


def get_all_chunks(
    conn: sqlite3.Connection,
    *,
    include_embedding: bool = False,
    only_with_embedding: bool = False,
) -> List[Dict[str, Any]]:
    """Return all chunks as list of dicts. Optionally include embedding and/or filter to embedded only."""
    cols = "c.chunk_id, c.doc_id, c.chunk_index, c.text, c.metadata_json, d.source_path, d.source_ext"
    if include_embedding or only_with_embedding:
        cols += ", c.embedding"
    q = f"""
        SELECT {cols}
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
    """
    if only_with_embedding:
        q += " WHERE c.embedding IS NOT NULL"
    q += " ORDER BY c.doc_id, c.chunk_index"
    cur = conn.execute(q)
    rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
        rec: Dict[str, Any] = {
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "chunk_index": r["chunk_index"],
            "text": r["text"],
            "metadata": meta,
            "doc_source": r["source_path"],
            "doc_type": (r["source_ext"] or "").lstrip("."),
        }
        if include_embedding or only_with_embedding:
            rec["embedding"] = _blob_to_embedding(r["embedding"])
        out.append(rec)
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


def update_chunk_embedding(
    conn: sqlite3.Connection,
    chunk_id: str,
    embedding: List[float],
) -> None:
    """Store embedding for one chunk (float32 BLOB)."""
    conn.execute(
        "UPDATE chunks SET embedding = ? WHERE chunk_id = ?",
        (_embedding_to_blob(embedding), chunk_id),
    )
    conn.commit()


def update_chunk_embeddings_batch(
    conn: sqlite3.Connection,
    items: List[tuple[str, List[float]]],
) -> None:
    """Store embeddings for multiple chunks. items = [(chunk_id, embedding), ...]."""
    for chunk_id, embedding in items:
        conn.execute(
            "UPDATE chunks SET embedding = ? WHERE chunk_id = ?",
            (_embedding_to_blob(embedding), chunk_id),
        )
    conn.commit()


def get_document_count(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM documents")
    return cur.fetchone()[0]


def get_chunk_count(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM chunks")
    return cur.fetchone()[0]


def get_chunk_count_with_embedding(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
    return cur.fetchone()[0]
