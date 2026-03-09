# Core logic for question-answering with Gemini and inline citations in a RAG system.
# This module defines data structures for citations and answer configuration, builds prompts for the Gemini API, calls the API, and extracts inline citations from the generated answer.
# Example usage:
#   answer = answer_question(
#       query="What is the distance to the nearest star?",
#       chunks=[{"chunk_id": "c1", "doc_source": "astronomy.txt", "metadata": {"page": 42}, "text": "The nearest star is Proxima Centauri, about 4.24 light-years away."}]
#   )
# print(answer.answer)  # The generated answer text with inline citations
# print(answer.citations)  # List of Citation objects with metadata about the cited chunks


from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1"


@dataclass
# --------- #
# -- Citation: dataclass for inline citation metadata -- #
# --------- #
class Citation:
    label: str
    chunk_id: str
    doc_source: Optional[str] = None
    page: Optional[int] = None


@dataclass
# --------- #
# -- AnswerWithCitations: answer text plus resolved citations -- #
# --------- #
class AnswerWithCitations:
    answer: str
    citations: List[Citation]


@dataclass
# --------- #
# -- AnswerConfig: generation model and output parameters -- #
# --------- #
class AnswerConfig:
    model: str = GEMINI_DEFAULT_MODEL
    temperature: float = 0.2
    max_output_tokens: int = 512


# --------- #
# -- _get_gemini_api_key: read and validate GEMINI API key from env -- #
# --------- #
def _get_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError("GEMINI_API_KEY is not set in environment/.env.")
    return key


# --------- #
# -- _call_gemini: send prompt to Gemini API and return generated text -- #
# --------- #
def _call_gemini(
    prompt: str,
    *,
    config: Optional[AnswerConfig] = None,
) -> str:
    cfg = config or AnswerConfig()
    api_key = _get_gemini_api_key()

    url = f"{GEMINI_API_BASE}/models/{cfg.model}:generateContent"

    body: Dict[str, Any] = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": cfg.temperature,
            "maxOutputTokens": cfg.max_output_tokens,
        },
    }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API HTTPError {e.code}: {msg}") from e
    except Exception as e:
        raise RuntimeError(f"Gemini API request failed: {e}") from e

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"Gemini API returned non-JSON response: {raw!r}")

    # Expected shape:
    # { "candidates": [ { "content": { "parts": [ { "text": "..."}, ... ] } } ] }
    cands = payload.get("candidates") or []
    if not cands:
        raise RuntimeError(f"Gemini API returned no candidates: {payload!r}")

    parts = (cands[0].get("content") or {}).get("parts") or []
    texts: List[str] = []
    for p in parts:
        t = p.get("text")
        if isinstance(t, str):
            texts.append(t)
    if not texts:
        raise RuntimeError(f"Gemini API candidate had no text parts: {payload!r}")
    return "".join(texts).strip()


_CIT_LABEL_RE = re.compile(r"\[(\d+)\]")


# --------- #
# -- _build_prompt: assemble model prompt from query and context chunks -- #
# --------- #
def _build_prompt(query: str, chunks: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(
        "You are an astronomy question-answering assistant for a Retrieval-Augmented Generation (RAG) system."
    )
    lines.append(
        "You must answer ONLY using the information contained in the numbered context chunks below."
    )
    lines.append(
        "If the context does not contain enough information to fully answer, say so clearly and do NOT invent facts."
    )
    lines.append(
        "When you use information from a chunk, add an inline citation like [1] or [2][3] referring to the chunk numbers."
    )
    lines.append(
        "Do not include a bibliography section; just use inline citations inside the answer."
    )
    lines.append("")
    lines.append("Context chunks:")
    for idx, c in enumerate(chunks, start=1):
        src = c.get("doc_source", "")
        meta = c.get("metadata", {}) or {}
        page = meta.get("page")
        header = f"[{idx}]"
        if src:
            header += f" source={src}"
        if page is not None:
            header += f" page={page}"
        lines.append(header)
        lines.append(c.get("text", ""))
        lines.append("")

    lines.append("User question:")
    lines.append(query.strip())
    lines.append("")
    lines.append(
        "Now write a concise answer grounded in the context with inline citations."
    )

    return "\n".join(lines)


# --------- #
# -- _extract_citations: parse inline [n] labels and map to chunks -- #
# --------- #
def _extract_citations(
    answer: str,
    chunks: Sequence[Dict[str, Any]],
) -> List[Citation]:
    seen_labels: Dict[int, Citation] = {}
    for match in _CIT_LABEL_RE.finditer(answer):
        n_str = match.group(1)
        try:
            idx = int(n_str)
        except ValueError:
            continue
        if idx < 1 or idx > len(chunks):
            continue
        if idx in seen_labels:
            continue
        c = chunks[idx - 1]
        meta = c.get("metadata", {}) or {}
        seen_labels[idx] = Citation(
            label=f"[{idx}]",
            chunk_id=c.get("chunk_id", ""),
            doc_source=c.get("doc_source"),
            page=meta.get("page"),
        )
    # Preserve numeric order
    return [seen_labels[k] for k in sorted(seen_labels.keys())]


# --------- #
# -- answer_question: build prompt, call model, extract citations -- #
# --------- #
def answer_question(
    query: str,
    chunks: Sequence[Dict[str, Any]],
    *,
    config: Optional[AnswerConfig] = None,
) -> AnswerWithCitations:
    """
    Generate an answer grounded in the provided chunks, with inline [n] citations.
    """
    if not chunks:
        raise ValueError("answer_question called with no context chunks.")

    prompt = _build_prompt(query, chunks)
    answer = _call_gemini(prompt, config=config)
    citations = _extract_citations(answer, chunks)
    return AnswerWithCitations(answer=answer, citations=citations)

