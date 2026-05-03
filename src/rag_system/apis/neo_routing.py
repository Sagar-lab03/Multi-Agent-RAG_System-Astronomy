from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

# -------------------------
# REGEX SIGNALS
# -------------------------

_ID_RE = re.compile(r"\b\d{5,}\b")
_PAGE_RE = re.compile(r"\b(page|pg)\s+\d+\b")
_DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")

# Relative / natural date words
_RELATIVE_DATE_RE = re.compile(
    r"\b(today|tomorrow|yesterday|week|month|next|upcoming|recent)\b"
)

# Very light asteroid name heuristic (can expand later)
_NAME_RE = re.compile(
    r"\b(apophis|bennu|eros|ryugu|itokawa)\b", re.IGNORECASE
)

# -------------------------
# PHRASE SETS
# -------------------------

_LOOKUP_PHRASES = (
    "tell me about",
    "details of",
    "what is",
    "info on",
    "information about",
    "describe asteroid",
)

# IMPORTANT: removed generic "list"
_BROWSE_PHRASES = (
    "browse asteroids",
    "asteroid catalog",
    "all asteroids",
    "asteroid database",
    "next page",
    "previous page",
)

_FEED_PHRASES = (
    "today",
    "this week",
    "asteroids now",
    "near earth",
    "close to earth",
    "approaching earth",
    "passing by earth",
    "upcoming asteroids",
    "recent asteroids",
    "asteroid closest approach"
)

# -------------------------
# DECISION CLASS
# -------------------------

@dataclass(frozen=True)
class NeoEndpointDecision:
    endpoint: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {"endpoint": self.endpoint, "reason": self.reason}


# -------------------------
# ROUTER
# -------------------------

def route_neo_endpoint(query: str) -> NeoEndpointDecision:
    """
    Deterministic post-intent router for NEO only.

    Priority:
      1) neo_lookup  (specific object)
      2) neo_browse  (pagination / dataset navigation)
      3) neo_feed    (time-based / general queries)
    """

    q = (query or "").strip().lower()

    # -------------------------
    # SIGNAL FLAGS
    # -------------------------

    has_id = bool(_ID_RE.search(q))
    has_name = bool(_NAME_RE.search(q))
    has_lookup_phrase = any(p in q for p in _LOOKUP_PHRASES)

    has_page = bool(_PAGE_RE.search(q))
    has_browse_phrase = any(p in q for p in _BROWSE_PHRASES)

    has_date = bool(_DATE_RE.search(q))
    has_relative_date = bool(_RELATIVE_DATE_RE.search(q))
    has_feed_phrase = any(p in q for p in _FEED_PHRASES)

    # -------------------------
    # 1) LOOKUP (highest priority)
    # -------------------------

    if has_id:
        return NeoEndpointDecision(
            endpoint="neo_lookup",
            reason="Detected numeric NEO ID.",
        )

    if has_name:
        return NeoEndpointDecision(
            endpoint="neo_lookup",
            reason="Detected known asteroid name.",
        )

    if has_lookup_phrase and not (has_page or has_browse_phrase):
        return NeoEndpointDecision(
            endpoint="neo_lookup",
            reason="Lookup-style phrasing detected.",
        )

    # -------------------------
    # 2) BROWSE
    # -------------------------

    if has_page:
        return NeoEndpointDecision(
            endpoint="neo_browse",
            reason="Pagination intent detected (page N).",
        )

    if has_browse_phrase and not (has_date or has_relative_date):
        return NeoEndpointDecision(
            endpoint="neo_browse",
            reason="Explicit dataset browsing intent.",
        )

    # -------------------------
    # 3) FEED (default)
    # -------------------------

    if has_date:
        return NeoEndpointDecision(
            endpoint="neo_feed",
            reason="Specific date detected.",
        )

    if has_relative_date:
        return NeoEndpointDecision(
            endpoint="neo_feed",
            reason="Relative time expression detected.",
        )

    if has_feed_phrase:
        return NeoEndpointDecision(
            endpoint="neo_feed",
            reason="Near-earth/time-based query.",
        )

    return NeoEndpointDecision(
        endpoint="neo_feed",
        reason="No strong lookup or browse signal; defaulting to feed.",
    )