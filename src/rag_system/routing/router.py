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
# Returns canonicalized label; UNKNOWN from rules + LLM off → DOCUMENT_SEARCH

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from .intents import (
    ALL_INTENTS,
    APOD,
    API_INTENTS,
    DOCUMENT_SEARCH,
    DONKI,
    EONET,
    NEO,
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
    "what are",
    "explain",
    "how does",
    "how do",
    "how can",
    "why",
    "difference between",
    "compare",
    "vs",
    "definition",
    "causes",
    "meaning of",
    "derive",
    "tell me about",
    "tell me",
    "describe",
    "summarize",
    "overview",
]

# Explanatory / contrast phrasing: generic taxonomy hits (e.g. "asteroid") must not beat documents.
_COMPARISON_MARKERS = (
    "difference between",
    "what is the difference",
    "what are the differences",
    "compared to",
    "compare",
    " versus ",
)
_VS_WORD = re.compile(r"\bvs\b")

# Normalized phrases or label tokens that show the user meant this NASA *service*, not a textbook word.
_API_SERVICE_EVIDENCE: Dict[str, FrozenSet[str]] = {
    NEO: frozenset(
        {
            "neo",
            "near earth object",
            "near-earth object",
            "nasa neo",
        }
    ),
    APOD: frozenset(
        {
            "apod",
            "astronomy picture of the day",
        }
    ),
    DONKI: frozenset({"donki", "space weather database"}),
    EONET: frozenset({"eonet", "earth observatory natural event"}),
}


def _looks_like_explanatory_comparison(nq: str) -> bool:
    if any(m in nq for m in _COMPARISON_MARKERS):
        return True
    if _VS_WORD.search(nq):
        return True
    return False


def _explicit_api_service_request(nq: str, api_label: str, matched_kws: List[str]) -> bool:
    """True only if the query clearly names the API/tool, not just a related science term."""
    token = api_label.strip().lower()
    if token and re.search(rf"\b{re.escape(token)}\b", nq):
        return True
    allowed = _API_SERVICE_EVIDENCE.get(api_label, frozenset())
    for raw in matched_kws:
        nk = _normalize(raw)
        if nk in allowed:
            return True
    return False


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


def _intent_score_tie_break(label: str) -> int:
    """Higher value wins when scores are tied (prefer specific API over generic doc)."""
    if label in API_INTENTS:
        return 3
    if label == DOCUMENT_SEARCH:
        return 2
    return 0


@dataclass
class RouterConfig:
    # Soft bias toward indexed documents every query (RAG-first product).
    document_search_prior: float = 1.0
    # Minimum score for the winning label to be accepted (UNKNOWN otherwise, unless branches below).
    min_score: float = 1.0
    # If top two scores are too close, treat as ambiguous (then use tie / concept / doc-top rules).
    min_margin: float = 0.5
    # Bonus for data-seeking queries ("today", "latest", "show", ...)
    data_bonus: float = 0.75
    # Bonus for conceptual queries ("what is", "explain", ...)
    concept_bonus: float = 0.75
    # If query strongly looks conceptual, prefer document search over UNKNOWN
    prefer_document_on_concept: bool = True
    # If the top-scoring label is DOCUMENT_SEARCH but the margin check failed, still pick documents.
    prefer_document_when_top: bool = True
    # "Difference between X and Y" / "compare …" → DOCUMENT_SEARCH unless the query names the API (neo, apod, …).
    demote_api_on_comparison_without_service_token: bool = True


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

        scores[DOCUMENT_SEARCH] += self.config.document_search_prior

        # Pick best label (score desc, then API > DOCUMENT_SEARCH > UNKNOWN on ties).
        ordered = sorted(
            scores.items(),
            key=lambda kv: (kv[1], _intent_score_tie_break(kv[0])),
            reverse=True,
        )
        best_label, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else -1.0
        margin = best_score - second_score

        # Only treat as ambiguous when second place is *strictly* behind but within margin.
        # margin == 0 (true tie, e.g. doc prior 1.0 vs APOD keyword 1.0): sort order already
        # prefers API over DOCUMENT_SEARCH — must not fall through to UNKNOWN.
        ambiguous = (margin > 0) and (margin < self.config.min_margin)

        decision = UNKNOWN

        if best_score >= self.config.min_score and not ambiguous:
            decision = best_label
        elif best_score >= self.config.min_score and ambiguous and best_label in API_INTENTS:
            # Narrow lead but still an API keyword path — prefer API over UNKNOWN.
            decision = best_label
        else:
            if self.config.prefer_document_on_concept and concept_hits:
                decision = DOCUMENT_SEARCH
            elif self.config.prefer_document_when_top and best_label == DOCUMENT_SEARCH:
                decision = DOCUMENT_SEARCH
            else:
                decision = UNKNOWN

        if (
            self.config.demote_api_on_comparison_without_service_token
            and decision in API_INTENTS
            and _looks_like_explanatory_comparison(nq)
            and not _explicit_api_service_request(nq, decision, matched.get(decision, []))
        ):
            decision = DOCUMENT_SEARCH

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
    - If rules yield UNKNOWN: optional LLM router when enabled, else DOCUMENT_SEARCH (RAG default)
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

    def route_with_trace(self, query: str) -> Tuple[str, RouteTrace]:
        """
        Returns (final_intent, rule_phase_trace). Trace.decision is the rule-only outcome;
        final_intent may be DOCUMENT_SEARCH when rules were UNKNOWN and LLM fallback is off.
        """
        rule_label, trace = self.rule.route_with_trace(query)
        if rule_label != UNKNOWN:
            return rule_label, trace
        if self.enable_llm_fallback:
            if self._llm is None:
                self._llm = PromptRouter(specs=self.specs)
            return canonicalize_intent(self._llm.route(query)), trace
        return DOCUMENT_SEARCH, trace

    def route(self, query: str) -> str:
        return self.route_with_trace(query)[0]

