# Purpose: the “single source of truth” for allowed router outputs.
# Critical: your router must emit exactly one of these labels.
# Also provides canonicalize_intent() for safety (especially for LLM outputs).

from __future__ import annotations

from typing import Final, FrozenSet, List

# Router must output EXACTLY one of these labels.
DOCUMENT_SEARCH: Final[str] = "DOCUMENT_SEARCH"
APOD: Final[str] = "APOD"
NEO: Final[str] = "NEO"
DONKI: Final[str] = "DONKI"
EONET: Final[str] = "EONET"
UNKNOWN: Final[str] = "UNKNOWN"

ALL_INTENTS: Final[FrozenSet[str]] = frozenset(
    {DOCUMENT_SEARCH, APOD, NEO, DONKI, EONET, UNKNOWN}
)

API_INTENTS: Final[FrozenSet[str]] = frozenset({APOD, NEO, DONKI, EONET})


def is_valid_intent(label: str) -> bool:
    return label in ALL_INTENTS


def canonicalize_intent(label: str) -> str:
    """
    Map a possibly messy LLM output to a known intent if possible.
    """
    s = (label or "").strip().upper()
    # Remove surrounding punctuation/quotes
    s = s.strip("`\"' \t\r\n.,;:()[]{}")
    if s in ALL_INTENTS:
        return s
    return UNKNOWN

