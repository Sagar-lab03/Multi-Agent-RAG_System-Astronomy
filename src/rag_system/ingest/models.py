# Purpose: data models for ingestion.
# Contains:
# - Document: a file with one or more sections (source path + metadata + list of sections)
# - Section: a text block with metadata (PDF pages become sections)
# - Chunk: a section split into overlapping text blocks (text + metadata + stable ids)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Section:
    """
    A document can be represented as one or more sections.
    For PDFs we typically use one section per page (with page metadata).
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Document:
    doc_id: str
    source_path: str
    source_ext: str
    metadata: Dict[str, Any]
    sections: List[Section]


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any]

