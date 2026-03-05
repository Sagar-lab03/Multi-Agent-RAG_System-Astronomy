from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .embedding import get_embedder
from .search import Embedder, hybrid_search, lexical_search, semantic_search


@dataclass
class RetrieverConfig:
    """
    Configuration for the retriever agent.

    mode:
      - "semantic": embeddings only
      - "lexical": BM25 only
      - "hybrid": combine semantic + lexical (default)
    """

    mode: str = "hybrid"
    top_k: int = 6
    alpha: float = 0.6  # semantic weight for hybrid


def retrieve_context(
    conn: Any,
    query: str,
    config: Optional[RetrieverConfig] = None,
    *,
    embedder_factory: Optional[Callable[[], Embedder]] = None,
) -> List[Dict[str, Any]]:
    """
    Core retriever entry point.

    - Hides whether we're using semantic / lexical / hybrid.
    - Returns a list of chunk dicts with at least:
      chunk_id, text, score, doc_source, doc_type, metadata, ...
    """
    cfg = config or RetrieverConfig()
    mode = cfg.mode.lower()

    if mode == "lexical":
        return lexical_search(conn, query, top_k=cfg.top_k)

    # Modes that require an embedder
    ef = embedder_factory or get_embedder
    embedder = ef()

    if mode == "semantic":
        return semantic_search(conn, query, embedder, top_k=cfg.top_k)
    if mode == "hybrid":
        return hybrid_search(conn, query, embedder, top_k=cfg.top_k, alpha=cfg.alpha)

    raise ValueError(f"Unknown retriever mode: {cfg.mode}")


__all__ = ["RetrieverConfig", "retrieve_context"]

