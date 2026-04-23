# Orchestrated query path: route → dispatch → RAG (DOCUMENT_SEARCH) or stub message.

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_system.apis import extract_date_from_query, fetch_apod
from rag_system.ingest.store import get_connection, init_schema
from rag_system.qa.answerer import AnswerConfig, AnswerWithCitations, answer_question
from rag_system.qa.verifier import VerificationResult, verify_answer
from rag_system.retrieval import RetrieverConfig, retrieve_context
from rag_system.routing.dispatch import DispatchPlan, build_dispatch_plan
from rag_system.routing.intents import APOD, API_INTENTS, DOCUMENT_SEARCH, UNKNOWN
from rag_system.routing.router import ConstrainedRouter, RouteTrace, RouterConfig


@dataclass
class OrchestrationResult:
    """Outcome of one orchestrated user query."""

    intent: str
    plan: DispatchPlan
    trace: Optional[RouteTrace]
    note: Optional[str]
    context: Optional[List[Dict[str, Any]]]
    answer: Optional[AnswerWithCitations]
    verification: Optional[VerificationResult]
    api_payload: Optional[Dict[str, Any]]
    error: Optional[str]


def _note_for_non_rag_intent(intent: str) -> str:
    if intent in API_INTENTS:
        return (
            "API not implemented yet. "
            f"(Classified as {intent!r} — no live NASA call in this build.) "
            "Rephrase for document-backed questions or use `route.py --debug` to inspect routing."
        )
    if intent == UNKNOWN:
        return (
            "Intent UNKNOWN: no confident route. Try a clearer scientific question over your corpus, "
            "or NASA-style keywords (e.g. APOD, NEO, DONKI, EONET)."
        )
    return f"Intent {intent!r}: no RAG pipeline is defined for this label yet."


def run_orchestrated_query(
    query: str,
    db_path: Path,
    *,
    retriever: RetrieverConfig,
    verify: bool,
    skip_router: bool = False,
    llm_router_fallback: bool = False,
    router_config: Optional[RouterConfig] = None,
) -> OrchestrationResult:
    """
    Classify the query (unless skip_router), build a dispatch plan, then either
    run the DOCUMENT_SEARCH RAG stack or return a short note for other intents.
    """
    q = (query or "").strip()
    trace: Optional[RouteTrace] = None
    if skip_router:
        intent = DOCUMENT_SEARCH
    else:
        router = ConstrainedRouter(
            config=router_config,
            enable_llm_fallback=llm_router_fallback,
        )
        intent, trace = router.route_with_trace(q)

    plan = build_dispatch_plan(intent)

    if intent == APOD:
        try:
            q_date = extract_date_from_query(q)
            apod_payload = fetch_apod(date=q_date)
        except Exception as e:
            return OrchestrationResult(
                intent=intent,
                plan=plan,
                trace=trace,
                note=None,
                context=None,
                answer=None,
                verification=None,
                api_payload=None,
                error=f"APOD request failed: {e}",
            )
        return OrchestrationResult(
            intent=intent,
            plan=plan,
            trace=trace,
            note=None,
            context=None,
            answer=None,
            verification=None,
            api_payload=apod_payload,
            error=None,
        )

    if intent != DOCUMENT_SEARCH:
        return OrchestrationResult(
            intent=intent,
            plan=plan,
            trace=trace,
            note=_note_for_non_rag_intent(intent),
            context=None,
            answer=None,
            verification=None,
            api_payload=None,
            error=None,
        )

    conn = get_connection(db_path)
    init_schema(conn)
    try:
        context = retrieve_context(conn, q, retriever)
    except Exception as e:
        conn.close()
        return OrchestrationResult(
            intent=intent,
            plan=plan,
            trace=trace,
            note=None,
            context=None,
            answer=None,
            verification=None,
            api_payload=None,
            error=f"Retrieval failed: {e}",
        )
    conn.close()

    if not context:
        return OrchestrationResult(
            intent=intent,
            plan=plan,
            trace=trace,
            note=None,
            context=[],
            answer=None,
            verification=None,
            api_payload=None,
            error="No context chunks retrieved. Run ingest and retrieval index first.",
        )

    try:
        result = answer_question(q, context, config=AnswerConfig())
    except Exception as e:
        return OrchestrationResult(
            intent=intent,
            plan=plan,
            trace=trace,
            note=None,
            context=context,
            answer=None,
            verification=None,
            api_payload=None,
            error=f"Answer generation failed: {e}",
        )

    ver: Optional[VerificationResult] = None
    if verify:
        try:
            ver = verify_answer(q, result.answer, context)
        except Exception as e:
            print(f"[WARN] Verification failed: {e}", file=sys.stderr)
            ver = None

    return OrchestrationResult(
        intent=intent,
        plan=plan,
        trace=trace,
        note=None,
        context=context,
        answer=result,
        verification=ver,
        api_payload=None,
        error=None,
    )


__all__ = ["OrchestrationResult", "run_orchestrated_query"]
