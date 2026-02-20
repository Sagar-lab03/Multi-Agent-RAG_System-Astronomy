# Purpose: loaders for different file types.
# Contains:
# - load_document: load a single file into a Document with one or more Sections.
# - _load_text_file: load a text file
# - _load_html_file: load an HTML file
# - _load_pdf_file: load a PDF file

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import Document, Section
from .utils import sha256_file


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _guess_title_from_text(text: str) -> Optional[str]:
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # Markdown heading
        if s.startswith("#"):
            return s.lstrip("#").strip() or None
        # Plain first line
        return s[:200]
    return None


def _load_text_file(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding, errors="replace")


def _load_html_file(path: Path, encoding: str = "utf-8") -> tuple[str, Optional[str]]:
    html = path.read_text(encoding=encoding, errors="replace")
    title: Optional[str] = None

    # Prefer BeautifulSoup if installed (declared in requirements.txt)
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip() or None
        text = soup.get_text("\n")
        return text, title
    except Exception:
        # Fallback: naive tag stripping (keeps text, loses structure)
        import re

        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
        if title_match:
            title = re.sub(r"\s+", " ", title_match.group(1)).strip() or None

        text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text, title


def _load_pdf_file(path: Path) -> tuple[List[Section], Dict[str, Any]]:
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(str(path))
    sections: List[Section] = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        sections.append(Section(text=text, metadata={"page": i}))

    pdf_meta: Dict[str, Any] = {
        "page_count": len(reader.pages),
    }

    # Best-effort title
    try:
        docinfo = reader.metadata
        if docinfo and getattr(docinfo, "title", None):
            pdf_meta["title"] = str(docinfo.title)
    except Exception:
        pass

    return sections, pdf_meta


def load_document(path: Path, *, encoding: str = "utf-8") -> Document:
    """
    Load a single file into a Document with one or more Sections.

    - txt/md: single section
    - html: single section (title from <title> if available)
    - pdf: one section per page (metadata includes page number)
    """

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))

    ext = path.suffix.lower()
    doc_id = sha256_file(path)

    base_meta: Dict[str, Any] = {
        "ingested_at": _now_iso(),
        "source_name": path.name,
        "source_ext": ext,
        "source_size_bytes": path.stat().st_size,
    }

    title: Optional[str] = None
    sections: List[Section]
    extra_meta: Dict[str, Any] = {}

    if ext in {".txt", ".md"}:
        text = _load_text_file(path, encoding=encoding)
        title = _guess_title_from_text(text)
        sections = [Section(text=text, metadata={})]
    elif ext in {".html", ".htm"}:
        text, html_title = _load_html_file(path, encoding=encoding)
        title = html_title or _guess_title_from_text(text)
        sections = [Section(text=text, metadata={})]
    elif ext == ".pdf":
        sections, extra_meta = _load_pdf_file(path)
        title = extra_meta.get("title") or None
        if not title and sections:
            title = _guess_title_from_text(sections[0].text)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if title:
        base_meta["title"] = title

    base_meta.update(extra_meta)

    return Document(
        doc_id=doc_id,
        source_path=str(path),
        source_ext=ext,
        metadata=base_meta,
        sections=sections,
    )

