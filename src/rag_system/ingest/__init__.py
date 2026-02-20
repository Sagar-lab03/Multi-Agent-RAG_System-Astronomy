# Purpose: expose ingest package API.
# Contains:
# - Chunk: a section split into overlapping text blocks (text + metadata + stable ids)
# - Document: a file with one or more sections (source path + metadata + list of sections)
# - Section: a text block with metadata (PDF pages become sections)
# - chunk_document(): applies chunking per Section and carries metadata forward
# - Chunk, Document, Section: data models
# - chunk_document(): chunking logic
# - store helpers: get_connection, init_schema, get_all_chunks, get_chunks_by_doc_id, etc.

from .chunking import chunk_document
from .models import Chunk, Document, Section
from .store import (
    get_chunk_count,
    get_connection,
    get_document_count,
    get_all_chunks,
    get_chunks_by_doc_id,
    init_schema,
    replace_all,
)

__all__ = [
    "Chunk",
    "Document",
    "Section",
    "chunk_document",
    "get_all_chunks",
    "get_chunk_count",
    "get_chunks_by_doc_id",
    "get_connection",
    "get_document_count",
    "init_schema",
    "replace_all",
]

