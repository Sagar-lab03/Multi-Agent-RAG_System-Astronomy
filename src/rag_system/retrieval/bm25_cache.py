from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rag_system.ingest.store import get_all_chunks

from .bm25 import BM25Index


def _db_file_from_conn(conn: sqlite3.Connection) -> str:
    """
    Best-effort identify which SQLite file this connection uses.
    Used only for in-process caching.
    """
    try:
        cur = conn.execute("PRAGMA database_list;")
        rows = cur.fetchall()
        # row: (seq, name, file)
        for r in rows:
            if r[1] == "main":
                return str(r[2] or "")
    except Exception:
        pass
    return ""


@dataclass(frozen=True)
class CorpusFingerprint:
    chunk_count: int
    max_created_at: Optional[str]


def compute_corpus_fingerprint(conn: sqlite3.Connection) -> CorpusFingerprint:
    cur = conn.execute("SELECT COUNT(*) AS n, MAX(created_at) AS m FROM chunks;")
    row = cur.fetchone()
    n = int(row[0] or 0)
    m = row[1]
    return CorpusFingerprint(chunk_count=n, max_created_at=m)


class BM25Cache:
    """
    Cache BM25Index and chunk lookup tables for one SQLite DB file.
    Automatically rebuilds when the corpus fingerprint changes.
    """

    def __init__(self, db_file: str) -> None:
        self.db_file = db_file
        self._fingerprint: Optional[CorpusFingerprint] = None
        self._index: Optional[BM25Index] = None
        self._chunks_by_id: Dict[str, Dict[str, Any]] = {}

    def ensure_ready(self, conn: sqlite3.Connection) -> None:
        fp = compute_corpus_fingerprint(conn)
        if self._fingerprint == fp and self._index is not None:
            return

        chunks = get_all_chunks(conn, include_embedding=False, only_with_embedding=False)
        self._chunks_by_id = {c["chunk_id"]: c for c in chunks}
        self._index = BM25Index((c["chunk_id"], c["text"]) for c in chunks)
        self._fingerprint = fp

    def search(self, conn: sqlite3.Connection, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        self.ensure_ready(conn)
        if self._index is None:
            return []
        scored = self._index.search(query, top_k=top_k)
        out: List[Dict[str, Any]] = []
        for chunk_id, score in scored:
            base = self._chunks_by_id.get(chunk_id)
            if not base:
                continue
            rec = dict(base)
            rec["score"] = score
            rec["score_lexical"] = score
            out.append(rec)
        return out


# One cache per process; keyed by DB file path.
_GLOBAL: Dict[str, BM25Cache] = {}


def get_bm25_cache(conn: sqlite3.Connection) -> BM25Cache:
    db_file = _db_file_from_conn(conn)
    key = db_file or "<memory>"
    cache = _GLOBAL.get(key)
    if cache is None:
        cache = BM25Cache(db_file=key)
        _GLOBAL[key] = cache
    return cache

