# Embedding providers: local (sentence-transformers) and Hugging Face Inference API.

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import List, Optional, Union
from urllib.parse import quote

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_INFERENCE_URL = "https://api-inference.huggingface.co/models"


class LocalEmbedder:
    """
    Embed text using sentence-transformers locally (CPU-friendly).
    Same interface as HuggingFaceEmbedder: embed_one, embed_many.
    """

    def __init__(self, *, model: Optional[str] = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model or os.environ.get("EMBEDDING_MODEL", "") or DEFAULT_MODEL
        self._model = SentenceTransformer(self.model_name)

    def embed_one(self, text: str) -> List[float]:
        """Return embedding vector for one string."""
        if not (text or "").strip():
            raise ValueError("text must be non-empty")
        vec = self._model.encode(text.strip(), normalize_embeddings=True,convert_to_numpy=True)
        return vec.tolist()

    def embed_many(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Return list of embedding vectors (batch_size used for encoding)."""
        cleaned = [t.strip() if (t or "").strip() else " " for t in texts]
        matrix = self._model.encode(cleaned, batch_size=batch_size, normalize_embeddings=True,convert_to_numpy=True)
        return [row.tolist() for row in matrix]


class HuggingFaceEmbedder:
    """
    Embed text via Hugging Face Inference API (serverless free tier).
    Uses sentence-transformers/all-MiniLM-L6-v2 by default (384-dim).
    """

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        retry_delay_s: float = 5.0,
    ) -> None:
        self.token = token or os.environ.get("HF_TOKEN", "").strip()
        if not self.token:
            raise ValueError(
                "Hugging Face token required. Set HF_TOKEN in .env or pass token=..."
            )
        self.model = model or os.environ.get("EMBEDDING_MODEL", "") or DEFAULT_MODEL
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        # Encode model id so path is correct (e.g. sentence-transformers%2Fall-MiniLM-L6-v2)
        self._base_url = f"{HF_INFERENCE_URL}/{quote(self.model, safe='')}"

    def _request(self, payload: dict) -> List[List[float]]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._base_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                out = json.loads(body)
                if isinstance(out, list) and len(out) > 0:
                    if isinstance(out[0], list):
                        return out
                    return [out]
                raise ValueError(f"Unexpected API response shape: {out!r}")
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code == 503:
                    # Model loading; wait and retry
                    time.sleep(self.retry_delay_s)
                    continue
                if e.code == 401:
                    raise ValueError("Invalid HF token (401). Check HF_TOKEN in .env.") from e
                if e.code == 410:
                    raise ValueError(
                        "Model/endpoint no longer available (410 Gone). "
                        "Try setting EMBEDDING_MODEL in .env to another model, e.g. "
                        "BAAI/bge-small-en-v1.5 or thenlper/gte-small, or use Hugging Face Inference Endpoints."
                    ) from e
                raise
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay_s)
                    continue
                raise
        raise last_err or RuntimeError("embedding request failed")

    def embed_one(self, text: str) -> List[float]:
        """Return embedding vector for one string."""
        if not (text or "").strip():
            raise ValueError("text must be non-empty")
        result = self._request({"inputs": text.strip()})
        return result[0]

    def embed_many(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Return list of embedding vectors. Batches requests to respect rate limits.
        """
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [t.strip() if (t or "").strip() else " " for t in texts[i : i + batch_size]]
            result = self._request({"inputs": batch})
            out.extend(result)
        return out


def get_embedder(
    *,
    backend: Optional[str] = None,
) -> Union["LocalEmbedder", "HuggingFaceEmbedder"]:
    """
    Return the embedder to use. Reads EMBEDDING_BACKEND from env:
    - "local" (default): LocalEmbedder (sentence-transformers)
    - "huggingface" or "hf": HuggingFaceEmbedder (Inference API)
    """
    backend = (backend or os.environ.get("EMBEDDING_BACKEND", "local")).strip().lower()
    if backend in ("huggingface", "hf"):
        return HuggingFaceEmbedder()
    return LocalEmbedder()
