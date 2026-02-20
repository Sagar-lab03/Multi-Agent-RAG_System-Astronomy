# Purpose: actual routing logic.
# It contains three router classes:
    
# 1) RuleBasedRouter (deterministic, debuggable)
# Normalizes query (_normalize)
# Scores intents based on phrase hits
# Adds “data-seeking” and “conceptual” bonuses
# Uses thresholds (RouterConfig) + ambiguity margin to decide
# Can return a full trace for debugging (RouteTrace)

# 2) PromptRouter (optional LLM-based)
# Calls an OpenAI-compatible /chat/completions endpoint
# Hard constraints: temperature=0, max_tokens=5
# Output is validated via canonicalize_intent(); failures → UNKNOWN

# 3) ConstrainedRouter (primary router - the “production” wrapper)
# Combines rule-based + prompt-based (if enabled)
# Only calls LLM if rule-based is UNKNOWN AND enabled
# Returns canonicalized label or UNKNOWN

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .intents import (
    ALL_INTENTS,
    API_INTENTS,
    DOCUMENT_SEARCH,
    UNKNOWN,
    canonicalize_intent,
)
from .specs import IntentSpec, default_intent_specs


_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_WS_RE = re.compile(r"\s+")


def _normalize(q: str) -> str:
    s = (q or "").strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _contains_any(normalized: str, phrases: List[str]) -> bool:
    for p in phrases:
        if not p:
            continue
        if p in normalized:
            return True
    return False


DATA_SIGNAL_PHRASES = [
    "today",
    "latest",
    "recent",
    "this week",
    "this month",
    "yesterday",
    "tomorrow",
    "last week",
    "current",
    "right now",
    "show",
    "list",
    "get",
    "fetch",
    "pull",
    "give me",
    "what happened",
    "events",
    "event",
    "alert",
    "alerts",
    "near me",
]

CONCEPT_SIGNAL_PHRASES = [
    "what is",
    "explain",
    "how does",
    "why",
    "difference between",
    "compare",
    "vs",
    "definition",
    "causes",
    "meaning of",
    "derive",
]


@dataclass(frozen=True)
class RouteTrace:
    """
    Debug-only trace: NEVER return this to the user if you require label-only output.
    """

    normalized_query: str
    scores: Dict[str, float]
    matched_keywords: Dict[str, List[str]]
    data_signals: List[str]
    concept_signals: List[str]
    decision: str


@dataclass
class RouterConfig:
    # Minimum score needed to choose a non-UNKNOWN label
    min_score: float = 2.0
    # If top two scores are too close, treat as ambiguous
    min_margin: float = 1.0
    # Bonus for data-seeking queries ("today", "latest", "show", ...)
    data_bonus: float = 0.75
    # Bonus for conceptual queries ("what is", "explain", ...)
    concept_bonus: float = 0.75
    # If query strongly looks conceptual, prefer document search over UNKNOWN
    prefer_document_on_concept: bool = True


class RuleBasedRouter:
    """
    Deterministic router. Explainable via RouteTrace.
    Output is always exactly ONE label from the known set.
    """

    def __init__(
        self,
        specs: Optional[Dict[str, IntentSpec]] = None,
        config: Optional[RouterConfig] = None,
    ) -> None:
        self.specs = specs or default_intent_specs()
        self.config = config or RouterConfig()

        # validate registry
        for label in self.specs.keys():
            if label not in ALL_INTENTS:
                raise ValueError(f"Unknown intent label in specs: {label}")

    def route_with_trace(self, query: str) -> Tuple[str, RouteTrace]:
        nq = _normalize(query)

        matched: Dict[str, List[str]] = {k: [] for k in self.specs.keys()}
        scores: Dict[str, float] = {k: 0.0 for k in self.specs.keys()}

        # Basic keyword scoring
        for label, spec in self.specs.items():
            if not spec.keywords:
                continue
            for kw in spec.keywords:
                nkw = _normalize(kw)
                if not nkw:
                    continue
                if nkw in nq:
                    matched[label].append(kw)
                    # Weighting: explicit API name tokens matter more
                    base = 2.0 if kw.strip().upper() == label else 1.0
                    scores[label] += base

        data_hits = [p for p in DATA_SIGNAL_PHRASES if p in nq]
        concept_hits = [p for p in CONCEPT_SIGNAL_PHRASES if p in nq]

        if data_hits:
            for api_label in API_INTENTS:
                scores[api_label] += self.config.data_bonus

        if concept_hits:
            scores[DOCUMENT_SEARCH] += self.config.concept_bonus

        # Pick best label
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_label, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else -1.0

        decision = UNKNOWN

        ambiguous = (best_score - second_score) < self.config.min_margin
        if best_score >= self.config.min_score and not ambiguous:
            decision = best_label
        else:
            # Safe fallback behavior
            if self.config.prefer_document_on_concept and concept_hits:
                decision = DOCUMENT_SEARCH
            else:
                decision = UNKNOWN

        trace = RouteTrace(
            normalized_query=nq,
            scores=scores,
            matched_keywords={k: v for k, v in matched.items() if v},
            data_signals=data_hits,
            concept_signals=concept_hits,
            decision=decision,
        )
        return decision, trace

    def route(self, query: str) -> str:
        label, _ = self.route_with_trace(query)
        # defensive
        return canonicalize_intent(label)


class PromptRouter:
    """
    Optional: prompt-based router using an OpenAI-compatible Chat Completions endpoint.
    Designed to be strictly constrained:
    - temperature=0
    - low max_tokens
    - output parsed/validated; invalid -> UNKNOWN
    """

    def __init__(
        self,
        *,
        specs: Optional[Dict[str, IntentSpec]] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 20.0,
    ) -> None:
        self.specs = specs or default_intent_specs()
        self.model = model or os.environ.get("ROUTER_LLM_MODEL", "")
        self.base_url = base_url or os.environ.get("ROUTER_LLM_BASE_URL", "")
        self.api_key = api_key or os.environ.get("ROUTER_LLM_API_KEY", "")
        self.timeout_s = timeout_s

        if not self.model or not self.base_url or not self.api_key:
            raise ValueError(
                "PromptRouter requires ROUTER_LLM_MODEL, ROUTER_LLM_BASE_URL, ROUTER_LLM_API_KEY (or explicit args)."
            )

    def _build_prompt(self, query: str) -> List[Dict[str, str]]:
        # Only descriptions for routing, not for answering.
        api_lines = []
        for label, spec in self.specs.items():
            if label in (DOCUMENT_SEARCH, UNKNOWN):
                continue
            api_lines.append(f"- {label}: {spec.short_description}")

        system = (
            "You are a query router. Decide which tool to route to.\n"
            "You MUST output exactly ONE label and nothing else.\n"
            "Valid labels: DOCUMENT_SEARCH, APOD, NEO, DONKI, EONET, UNKNOWN.\n"
            "If the intent is unclear or needs clarification, output UNKNOWN.\n"
            "Do not answer the user. Do not add explanations."
        )

        user = (
            "API descriptions (for routing only):\n"
            + "\n".join(api_lines)
            + "\n\nUser query:\n"
            + (query or "")
            + "\n\nOutput exactly one label."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def route(self, query: str) -> str:
        payload = {
            "model": self.model,
            "messages": self._build_prompt(query),
            "temperature": 0,
            "max_tokens": 5,
        }

        url = self.base_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return canonicalize_intent(content)
        except Exception:
            return UNKNOWN


class ConstrainedRouter:
    """
    Primary router:
    - Deterministic rule-based routing first
    - Optional prompt-based routing only if enabled AND rule-based is UNKNOWN
    """

    def __init__(
        self,
        *,
        specs: Optional[Dict[str, IntentSpec]] = None,
        config: Optional[RouterConfig] = None,
        enable_llm_fallback: bool = False,
    ) -> None:
        self.specs = specs or default_intent_specs()
        self.rule = RuleBasedRouter(specs=self.specs, config=config)
        self.enable_llm_fallback = enable_llm_fallback
        self._llm: Optional[PromptRouter] = None

        if enable_llm_fallback:
            # Lazy init on first need to avoid env-var requirements in default flow.
            self._llm = None

    def route(self, query: str) -> str:
        label = self.rule.route(query)
        if label != UNKNOWN:
            return label
        if not self.enable_llm_fallback:
            return UNKNOWN

        if self._llm is None:
            self._llm = PromptRouter(specs=self.specs)
        return canonicalize_intent(self._llm.route(query))

    def route_with_trace(self, query: str) -> Tuple[str, RouteTrace]:
        # Trace is from deterministic phase (LLM fallback is intentionally not traced here).
        return self.rule.route_with_trace(query)

