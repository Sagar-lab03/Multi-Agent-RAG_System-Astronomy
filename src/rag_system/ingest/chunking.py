# Purpose: split document text into retrieval-sized chunks with overlap.
# Contains:
# normalize_text(): standardizes newlines and collapses excessive blank lines
# _find_best_break(): tries to end chunks on “nice” boundaries (paragraph, newline, sentence-ish, whitespace)
# - chunk_text(): does the sliding window + overlap mechanics
# - chunk_document(): applies chunking per Section and carries metadata forward

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import Chunk, Document, Section


_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """
    Minimal normalization to make chunk boundaries more stable:
    - normalize newlines
    - collapse excessive blank lines
    - trim trailing spaces
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


def _find_best_break(text: str, start: int, proposed_end: int, lookback: int = 300) -> int:
    """
    Move end leftwards to a nicer boundary within a lookback window.
    Priority: paragraph -> newline -> sentence -> whitespace.
    """
    if proposed_end >= len(text):
        return len(text)

    window_start = max(start, proposed_end - lookback)
    window = text[window_start:proposed_end]

    # Prefer paragraph boundary
    idx = window.rfind("\n\n")
    if idx != -1 and window_start + idx > start:
        return window_start + idx

    # Then single newline
    idx = window.rfind("\n")
    if idx != -1 and window_start + idx > start:
        return window_start + idx

    # Then sentence-ish boundary
    # (This is intentionally simple to avoid heavy dependencies.)
    for pat in [". ", "? ", "! "]:
        idx = window.rfind(pat)
        if idx != -1 and window_start + idx + 2 > start:
            return window_start + idx + 2

    # Fallback to whitespace
    idx = window.rfind(" ")
    if idx != -1 and window_start + idx > start:
        return window_start + idx

    return proposed_end


def chunk_text(
    text: str,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> List[Tuple[int, int, str]]:
    """
    Return list of (char_start, char_end, chunk_text) on normalized text.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be < max_chars")

    text = normalize_text(text)
    if not text:
        return []

    chunks: List[Tuple[int, int, str]] = []
    start = 0

    while start < len(text):
        proposed_end = min(start + max_chars, len(text))
        end = _find_best_break(text, start, proposed_end)
        if end <= start:
            end = proposed_end

        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))

        if end >= len(text):
            break

        # next window starts with overlap
        next_start = max(end - overlap_chars, 0)
        if next_start <= start:
            next_start = min(start + 1, len(text))
        start = next_start

    return chunks


def chunk_document(
    doc: Document,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> List[Chunk]:
    """
    Chunk each section independently (PDF pages stay page-grounded).
    """
    out: List[Chunk] = []
    chunk_index = 0

    for section_idx, section in enumerate(doc.sections):
        normalized = normalize_text(section.text)
        if not normalized:
            continue

        for start, end, chunk_text_str in chunk_text(
            normalized, max_chars=max_chars, overlap_chars=overlap_chars
        ):
            meta: Dict[str, Any] = {
                **doc.metadata,
                **section.metadata,
                "section_index": section_idx,
                "char_start": start,
                "char_end": end,
                "chunk_chars": len(chunk_text_str),
            }

            chunk_id = f"{doc.doc_id}:{chunk_index}"
            out.append(
                Chunk(
                    doc_id=doc.doc_id,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    text=chunk_text_str,
                    metadata=meta,
                )
            )
            chunk_index += 1

    return out

