from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]


class BM25Index:
    """
    Simple BM25 index over chunks.
    Each "document" is a chunk; we index its text and keep track of chunk metadata externally.
    """

    def __init__(
        self,
        docs: Iterable[Tuple[str, str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """
        docs: iterable of (chunk_id, text)
        """
        self.k1 = k1
        self.b = b

        self.doc_ids: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lens: List[int] = []

        # term -> doc frequency
        self.df: Dict[str, int] = defaultdict(int)

        for chunk_id, text in docs:
            tokens = tokenize(text)
            self.doc_ids.append(chunk_id)
            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens))
            # update df once per doc
            seen = set(tokens)
            for t in seen:
                self.df[t] += 1

        self.N = len(self.doc_ids)
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 0.0
        # precompute idf
        self.idf: Dict[str, float] = {}
        for term, df in self.df.items():
            # classic BM25 idf
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score_doc(self, doc_index: int, query_terms: List[str]) -> float:
        tokens = self.doc_tokens[doc_index]
        if not tokens:
            return 0.0
        tf = Counter(tokens)
        score = 0.0
        dl = self.doc_lens[doc_index]
        for term in query_terms:
            if term not in tf or term not in self.idf:
                continue
            freq = tf[term]
            idf = self.idf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            score += idf * (freq * (self.k1 + 1) / denom)
        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Return list of (chunk_id, score) sorted by score desc.
        """
        terms = tokenize(query)
        if not terms or self.N == 0:
            return []
        scores: List[Tuple[float, str]] = []
        for i, chunk_id in enumerate(self.doc_ids):
            s = self.score_doc(i, terms)
            if s > 0:
                scores.append((s, chunk_id))
        scores.sort(key=lambda x: -x[0])
        return [(cid, s) for s, cid in scores[:top_k]]

