# Purpose: utility functions for ingestion.
# Contains:
# - sha256_file: hash a file
# - sha256_text: hash a text string
# Why:
# doc_id should be stable across runs → it’s computed as sha256(file bytes)
# later, stable ids make caching/debugging/eval much easier

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

