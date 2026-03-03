# Vector search: embed query, cosine similarity, top-k.

from __future__ import annotations

import math
from typing import Any, Dict, List, Protocol

from rag_system.ingest.store import get_all_chunks

from .bm25_cache import get_bm25_cache


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class Embedder(Protocol):
    def embed_one(self, text: str) -> List[float]: ...


def semantic_search(
    conn: Any,
    query: str,
    embedder: Embedder,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Semantic search: cosine similarity over stored embeddings.
    """
    query_embedding = embedder.embed_one(query)
    chunks = get_all_chunks(conn, include_embedding=True, only_with_embedding=True)
    if not chunks:
        return []

    scored: List[tuple[float, Dict[str, Any]]] = []
    for c in chunks:
        emb = c.get("embedding")
        if not emb or len(emb) != len(query_embedding):
            continue
        score = cosine_similarity(query_embedding, emb)
        scored.append((score, {**c, "score": score, "score_semantic": score}))

    scored.sort(key=lambda x: -x[0])
    results = [item[1] for item in scored[:top_k]]
    for r in results:
        r.pop("embedding", None)
    return results


def lexical_search(
    conn: Any,
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Lexical search: BM25 over chunk text.
    """
    cache = get_bm25_cache(conn)
    return cache.search(conn, query, top_k=top_k)


def hybrid_search(
    conn: Any,
    query: str,
    embedder: Embedder,
    top_k: int = 5,
    alpha: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Hybrid search: combine semantic and lexical scores with a weighted sum.
    alpha: weight for semantic (0–1); (1-alpha) for lexical.
    """
    sem = semantic_search(conn, query, embedder, top_k=top_k * 2)
    lex = lexical_search(conn, query, top_k=top_k * 2)

    by_id: Dict[str, Dict[str, Any]] = {}
    for r in sem:
        by_id.setdefault(r["chunk_id"], {}).update(r)
    for r in lex:
        by_id.setdefault(r["chunk_id"], {}).update(r)

    # Normalize lexical scores to [0,1] within this query
    lex_scores = [v.get("score_lexical", 0.0) for v in by_id.values()]
    max_lex = max(lex_scores) if lex_scores else 0.0

    results: List[Dict[str, Any]] = []
    for chunk_id, rec in by_id.items():
        s_sem = float(rec.get("score_semantic", 0.0))
        raw_lex = float(rec.get("score_lexical", 0.0))
        s_lex = raw_lex / max_lex if max_lex > 0 else 0.0
        hybrid = alpha * s_sem + (1 - alpha) * s_lex
        out = dict(rec)
        out["score"] = hybrid
        out["score_semantic"] = s_sem
        out["score_lexical"] = raw_lex
        results.append(out)

    results.sort(key=lambda r: -r["score"])
    # limit and drop any embedding field if present
    trimmed = results[:top_k]
    for r in trimmed:
        r.pop("embedding", None)
    return trimmed

