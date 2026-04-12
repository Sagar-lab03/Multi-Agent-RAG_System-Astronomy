"""
Streamlit debug UI for RAG QA: optional router → retrieval → Gemini answer → optional verification.
Run from project root: streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

# Project root = parent of ui/
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

import streamlit as st

from rag_system.orchestration import OrchestrationResult, run_orchestrated_query
from rag_system.retrieval import RetrieverConfig
from rag_system.routing.intents import API_INTENTS, DOCUMENT_SEARCH, UNKNOWN
from rag_system.routing.router import RouteTrace


def _sorted_scores(scores: Dict[str, float]) -> List[tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _render_why_from_trace(trace: RouteTrace) -> None:
    """Human-readable summary of the rule-router trace (scores, hits, signals)."""
    st.markdown(f"**Normalized query:** `{trace.normalized_query}`")
    st.markdown(f"**Rule decision (pre–LLM fallback):** `{trace.decision}`")

    top = _sorted_scores(trace.scores)[:6]
    lines = [f"- `{lbl}`: **{sc:.2f}**" for lbl, sc in top]
    st.markdown("**Intent scores (top):**\n" + "\n".join(lines))

    matched_any = {k: v for k, v in trace.matched_keywords.items() if v}
    if matched_any:
        st.markdown("**Keyword hits:**")
        for intent, kws in sorted(matched_any.items(), key=lambda kv: -trace.scores.get(kv[0], 0)):
            st.caption(f"{intent}: {', '.join(kws)}")
    else:
        st.caption("No intent keyword hits.")

    if trace.data_signals:
        st.markdown(f"**Data-seeking signals matched:** {', '.join(trace.data_signals)}")
    if trace.concept_signals:
        st.markdown(f"**Concept signals matched:** {', '.join(trace.concept_signals)}")


def _render_router_panel(
    use_router: bool,
    orch: OrchestrationResult,
    *,
    llm_router_fallback_enabled: bool,
) -> None:
    """Always show after a run: label, dispatch, and why (trace) when routing ran."""
    st.subheader("Router")
    if not use_router:
        st.caption(
            "Router was **skipped** — pipeline is forced to **DOCUMENT_SEARCH**. "
            "No intent trace."
        )
        st.markdown(f"**Effective label:** `{DOCUMENT_SEARCH}`")
        return

    st.markdown(f"**Label:** `{orch.intent}`")
    st.markdown(
        f"**Dispatch:** `should_call_api={orch.plan.should_call_api}` · `api_name={orch.plan.api_name!r}`"
    )
    if (
        orch.trace is not None
        and orch.trace.decision == UNKNOWN
        and orch.intent == DOCUMENT_SEARCH
        and not llm_router_fallback_enabled
    ):
        st.info(
            "Rule phase ended in **UNKNOWN**, but **LLM router fallback is off** — the pipeline uses "
            "**DOCUMENT_SEARCH** (indexed documents) as the safe default."
        )
    if orch.trace is not None:
        st.markdown("**Why (rule phase):**")
        _render_why_from_trace(orch.trace)
        with st.expander("Raw route trace (JSON)", expanded=False):
            st.json(asdict(orch.trace))
    else:
        st.caption("No route trace available.")


def main() -> None:
    st.set_page_config(page_title="RAG QA (debug)", layout="wide")
    st.title("Multi-Agent RAG — QA debug console")
    st.caption(
        "Optional router → dispatch → DOCUMENT_SEARCH uses hybrid/semantic/lexical retrieval + Gemini + "
        "optional verifier. NASA API intents are detected but not called yet."
    )

    with st.sidebar:
        st.header("Settings")
        db_default = str(_ROOT / "data" / "processed" / "chunks.db")
        db_path = st.text_input("SQLite chunk DB", value=db_default)
        use_router = st.checkbox("Use router (multi-agent path)", value=True)
        llm_route = st.checkbox(
            "LLM router fallback if rules say UNKNOWN",
            value=False,
            disabled=not use_router,
            help="OpenAI-compatible chat/completions endpoint (ROUTER_LLM_* env vars).",
        )
        mode = st.selectbox("Retriever mode", ["hybrid", "semantic", "lexical"], index=0)
        top_k = st.number_input("Top-k chunks", min_value=1, max_value=30, value=6)
        alpha = st.slider("Hybrid α (semantic weight)", 0.0, 1.0, 0.6, 0.05)
        run_verify = st.checkbox("Run verification (Gemini)", value=True)
        st.divider()
        st.markdown("Run from repo root: `streamlit run ui/streamlit_app.py`")

    query = st.text_area("Question", height=100, placeholder="e.g. What is gravitational redshift?")
    submitted = st.button("Run pipeline", type="primary")

    if not submitted:
        return
    q = (query or "").strip()
    if not q:
        st.warning("Enter a question.")
        return

    path = Path(db_path)
    if not path.exists():
        st.error(f"Database not found: {path}")
        return

    cfg = RetrieverConfig(mode=mode, top_k=int(top_k), alpha=float(alpha))
    with st.spinner("Running pipeline…"):
        orch = run_orchestrated_query(
            q,
            path,
            retriever=cfg,
            verify=run_verify,
            skip_router=not use_router,
            llm_router_fallback=llm_route and use_router,
        )

    _render_router_panel(use_router, orch, llm_router_fallback_enabled=llm_route and use_router)

    if orch.note:
        if orch.intent in API_INTENTS:
            st.warning("**API not implemented yet**")
            st.info(orch.note)
        else:
            st.info(orch.note)
        return

    if orch.error:
        st.error(orch.error)
        if orch.context is not None:
            with st.expander("Retrieved context (partial / before answer failed)", expanded=True):
                for c in orch.context:
                    st.text((c.get("text") or "")[:1500])
                    st.divider()
        return

    assert orch.answer is not None
    st.subheader("Answer")
    st.markdown(orch.answer.answer)

    st.subheader("Sources")
    if not orch.answer.citations:
        st.info("No inline citations detected in the answer.")
    else:
        for c in orch.answer.citations:
            src = c.doc_source or ""
            page = f", page **{c.page}**" if c.page is not None else ""
            st.markdown(f"- **{c.label}** `chunk_id={c.chunk_id}` — `{src}`{page}")

    if orch.verification is not None:
        st.subheader("Verification")
        v = orch.verification
        col1, col2 = st.columns(2)
        col1.metric("Grounded", "yes" if v.is_grounded else "no")
        col2.metric("Complete", "yes" if v.is_complete else "no")
        if v.issues:
            for i in v.issues:
                labels = ", ".join(i.citation_labels) if i.citation_labels else "—"
                st.warning(f"**{i.type}** (citations: {labels})\n\n{i.description}")
        else:
            st.success("No issues reported.")

    with st.expander("Retrieved context (chunks)", expanded=False):
        for idx, c in enumerate(orch.context or [], start=1):
            score = c.get("score", 0)
            src = c.get("doc_source", "")
            meta = c.get("metadata") or {}
            page = meta.get("page")
            st.markdown(f"**[{idx}]** score={score:.4f} · `{src}`" + (f" · page {page}" if page is not None else ""))
            st.text((c.get("text") or "")[:2000] + ("…" if len(c.get("text") or "") > 2000 else ""))
            st.divider()


if __name__ == "__main__":
    main()
