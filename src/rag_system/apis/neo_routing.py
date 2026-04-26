from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

_ID_RE = re.compile(r"\b\d{5,}\b")
_PAGE_RE = re.compile(r"\bpage\s+\d+\b")
_DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")

_LOOKUP_PHRASES = (
    "tell me about asteroid",
    "asteroid details",
    "info about asteroid",
)
_BROWSE_PHRASES = (
    "list",
    "browse",
    "catalog",
    "all asteroids",
    "next page",
    "previous page",
)
_FEED_PHRASES = (
    "today",
    "near earth",
    "asteroids now",
    "asteroid closest approach",
)


@dataclass(frozen=True)
class NeoEndpointDecision:
    endpoint: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {"endpoint": self.endpoint, "reason": self.reason}


def route_neo_endpoint(query: str) -> NeoEndpointDecision:
    """
    Deterministic post-intent router for NEO only.
    Priority:
      1) neo_lookup
      2) neo_browse
      3) neo_feed (default)
    """
    q = (query or "").strip().lower()

    # 1) LOOKUP (highest priority)
    if _ID_RE.search(q):
        return NeoEndpointDecision(
            endpoint="neo_lookup",
            reason="Query contains a likely NEO numeric id (5+ digits).",
        )
    if any(p in q for p in _LOOKUP_PHRASES):
        return NeoEndpointDecision(
            endpoint="neo_lookup",
            reason="Query requests details/info for a specific asteroid.",
        )

    # 2) BROWSE
    if _PAGE_RE.search(q):
        return NeoEndpointDecision(
            endpoint="neo_browse",
            reason="Query indicates pagination intent (page N).",
        )
    if any(p in q for p in _BROWSE_PHRASES):
        return NeoEndpointDecision(
            endpoint="neo_browse",
            reason="Query asks to list/browse/catalog asteroids.",
        )

    # 3) FEED (default fallback)
    if _DATE_RE.search(q):
        return NeoEndpointDecision(
            endpoint="neo_feed",
            reason="Query includes a date, which maps to feed-style date windows.",
        )
    if any(p in q for p in _FEED_PHRASES):
        return NeoEndpointDecision(
            endpoint="neo_feed",
            reason="Query is a general near-earth/current asteroid request.",
        )
    return NeoEndpointDecision(
        endpoint="neo_feed",
        reason="No explicit lookup or browse signal; defaulting to feed.",
    )

