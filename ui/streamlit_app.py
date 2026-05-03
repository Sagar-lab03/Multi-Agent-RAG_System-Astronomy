"""
Streamlit debug UI for RAG QA: optional router → retrieval → Gemini answer → optional verification.
Run from project root: streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

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

from rag_system.apis.neo_lookup import extract_asteroid_id_from_query, fetch_and_process_neo_lookup
from rag_system.orchestration import OrchestrationResult, run_orchestrated_query
from rag_system.retrieval import RetrieverConfig
from rag_system.routing.intents import APOD, API_INTENTS, DOCUMENT_SEARCH, NEO, UNKNOWN
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


def _render_apod_card(payload: Dict[str, object]) -> None:
    st.subheader("APOD")
    title = str(payload.get("title") or "(untitled)")
    date = str(payload.get("date") or "")
    media_type = str(payload.get("media_type") or "")
    image_url = str(payload.get("url") or "")
    hd_url = str(payload.get("hdurl") or "")
    explanation = str(payload.get("explanation") or "")

    st.markdown(f"### {title}")
    if date:
        st.caption(f"Date: {date}")
    if media_type:
        st.caption(f"Media type: {media_type}")

    if media_type == "image" and image_url:
        st.image(image_url, caption=title, use_container_width=True)
    elif media_type == "video" and image_url:
        st.video(image_url)
    elif image_url:
        st.markdown(f"[Open media URL]({image_url})")

    if explanation:
        st.markdown(explanation)

    if hd_url:
        st.markdown(f"[HD image]({hd_url})")
    if image_url and image_url != hd_url:
        st.markdown(f"[Media URL]({image_url})")


def _render_neo_feed(payload: Dict[str, object]) -> None:
    start_date = str(payload.get("start_date") or "")
    end_date = str(payload.get("end_date") or "")
    asteroids = payload.get("asteroids") or []
    if not isinstance(asteroids, list):
        asteroids = []

    st.subheader(f"🚀 Near-Earth Objects ({start_date} to {end_date})")

    total = int(payload.get("asteroid_count") or 0)
    hazardous = int(payload.get("hazardous_count") or 0)
    safe = int(payload.get("safe_count") or max(0, total - hazardous))
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Asteroids", total)
    c2.metric("Hazardous", hazardous)
    c3.metric("Safe", safe)
    st.divider()

    if not asteroids:
        st.info("No asteroids found for this date range.")
        return

    for idx, asteroid in enumerate(asteroids, start=1):
        with st.container():
            name = str(asteroid.get("name") or "Unknown")
            asteroid_id = str(asteroid.get("id") or "")
            hazardous_flag = bool(asteroid.get("is_potentially_hazardous_asteroid", False))
            diameter = str(asteroid.get("diameter_display") or "N/A")
            velocity = str(asteroid.get("velocity_display") or "N/A")
            miss_distance = str(asteroid.get("miss_distance_display") or "N/A")

            st.markdown(f"**{name}**")
            if hazardous_flag:
                st.markdown("**:red[⚠️ Hazardous]**")
            else:
                st.markdown("**:green[🟢 Safe]**")

            st.caption(f"Diameter: {diameter}")
            st.caption(f"Relative Velocity: {velocity}")
            st.caption(f"Miss Distance: {miss_distance}")

            if st.button("View Details", key=f"neo_view_{asteroid_id}_{idx}"):
                st.session_state.selected_asteroid_id = asteroid_id
                st.session_state.last_feed_data = payload
                st.session_state.view = "lookup"
                st.rerun()
        st.divider()


def _neo_init_session_defaults() -> None:
    defaults: Dict[str, Any] = {
        "selected_asteroid_id": None,
        "last_feed_data": None,
        "current_page": None,
        "view": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_data(ttl=300)
def _cached_neo_lookup(asteroid_id: str) -> Dict[str, Any]:
    """Cache NeoWs lookups for responsive drill-down."""
    return fetch_and_process_neo_lookup(asteroid_id)


def _neo_navigation_back_from_lookup() -> None:
    """Return to cached feed/browse without refetch."""
    if st.session_state.get("last_feed_data") is not None:
        st.session_state.view = "feed"
    elif st.session_state.get("current_page") is not None:
        st.session_state.view = "browse"
    else:
        st.session_state.view = "feed"


def _neo_render_lookup_back_bar(*, button_key_prefix: str) -> None:
    if st.button("← Back", key=f"{button_key_prefix}_neo_back_lookup"):
        _neo_navigation_back_from_lookup()
        st.rerun()


def _neo_render_lookup_body(struct: Dict[str, object]) -> None:
    """Drill-down card (no JSON), shared by pipeline lookup and standalone lookup."""
    name = str(struct.get("name") or "(unknown name)")
    jpl = str(struct.get("nasa_jpl_url") or "")
    hazardous = bool(struct.get("is_potentially_hazardous_asteroid", False))
    diameter_display = str(struct.get("diameter_display") or "N/A")
    velocity_display = str(struct.get("velocity_display") or "N/A")
    miss_display = str(struct.get("miss_distance_display") or "N/A")
    orbit_class = str(struct.get("orbit_class") or "N/A")
    orbiting_body = str(struct.get("orbiting_body") or "N/A")
    cad_date = str(struct.get("close_approach_date") or "N/A")

    st.markdown(f"### **{name}**")
    if hazardous:
        st.markdown("**:red[⚠️ Hazardous]**")
    else:
        st.markdown("**:green[🟢 Not Hazardous]**")

    lc, rc = st.columns(2)
    with lc:
        st.markdown("**Diameter**")
        st.caption(diameter_display)
        st.markdown("**Velocity**")
        st.caption(velocity_display)
        st.markdown("**Miss distance**")
        st.caption(miss_display)
    with rc:
        st.markdown("**Orbit class**")
        st.caption(orbit_class)
        st.markdown("**Orbiting body**")
        st.caption(orbiting_body)
        st.markdown("**Close approach date**")
        st.caption(cad_date)

    if jpl:
        st.markdown(f"[View on NASA]({jpl})")


def _neo_try_render_lookup_standalone(query_fallback: str) -> None:
    """Interactive lookup driven by session + optional ID in query text."""
    _neo_render_lookup_back_bar(button_key_prefix="standalone")

    primary = str(st.session_state.get("selected_asteroid_id") or "").strip()
    asteroid_id = primary or (extract_asteroid_id_from_query(query_fallback) or "")

    if not asteroid_id:
        st.warning("No asteroid selected. Please choose an asteroid from the list.")
        st.session_state.view = "feed"
        feed = st.session_state.get("last_feed_data")
        if feed is not None:
            _render_neo_feed(feed)
        else:
            st.caption("Run a NEO feed query first, then pick **View Details** on an asteroid.")
        return

    try:
        struct = _cached_neo_lookup(asteroid_id)
    except Exception:
        st.error("Asteroid not found or invalid ID.")
        if st.button("← Back", key="neo_lookup_err_back"):
            _neo_navigation_back_from_lookup()
            st.rerun()
        return

    _neo_render_lookup_body(struct)


def main() -> None:
    st.set_page_config(page_title="RAG QA (debug)", layout="wide")
    
    # Title with logo
    col1, col2 = st.columns([5, 1], gap="large")
    with col1:
        st.title("Multi-Agent RAG — AstroMind QA Lab")
    with col2:
        # Add your logo image here - replace with your actual logo path
        logo_path = _ROOT / "ui" / "image.jpg"  # Update this path to your logo file
        if logo_path.exists():
            st.image(str(logo_path), width=100, use_container_width=False)
    
    st.caption(
        "Optional router dispatches queries to DOCUMENT_SEARCH using hybrid (semantic + lexical) "
        "retrieval with Gemini and an optional verifier."
    )
    st.caption(
        "APOD and NEO (feed + lookup) are live; other NASA endpoints are placeholders."
    )

    _neo_init_session_defaults()

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
        show_router_panel = st.checkbox(
            "Show router debug section",
            value=True,
            help="If off, hides router/dispatch debug details from the main UI.",
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
        view = st.session_state.get("view")
        if view == "lookup":
            _neo_try_render_lookup_standalone(query or "")
            return
        if view == "feed" and st.session_state.get("last_feed_data") is not None:
            _render_neo_feed(st.session_state.last_feed_data)
            return
        if view == "browse":
            st.info("Browse view coming soon.")
            return
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

    if show_router_panel:
        with st.expander("Router details", expanded=False):
            _render_router_panel(use_router, orch, llm_router_fallback_enabled=llm_route and use_router)

    if orch.error:
        st.error(orch.error)
        if orch.context is not None:
            with st.expander("Retrieved context (partial / before answer failed)", expanded=True):
                for c in orch.context:
                    st.text((c.get("text") or "")[:1500])
                    st.divider()
        if orch.intent == NEO:
            if st.button("← Back", key="neo_pipeline_error_back"):
                _neo_navigation_back_from_lookup()
                st.rerun()
        return

    if orch.intent == APOD and orch.api_payload is not None:
        st.session_state.view = None
        _render_apod_card(orch.api_payload)
        return

    if orch.intent == NEO and orch.api_payload is not None:
        endpoint = str(orch.api_payload.get("endpoint") or "")
        if endpoint == "neo_feed":
            st.session_state.last_feed_data = orch.api_payload
            st.session_state.view = "feed"
            _render_neo_feed(orch.api_payload)
            return
        if endpoint == "neo_lookup":
            lid = str(orch.api_payload.get("id") or "").strip()
            if lid:
                st.session_state.selected_asteroid_id = lid
            st.session_state.view = "lookup"
            _neo_render_lookup_back_bar(button_key_prefix="pipeline_lookup")
            _neo_render_lookup_body(orch.api_payload)
            return

    if orch.note:
        if orch.intent in API_INTENTS:
            st.warning("**API not implemented yet**")
            st.info(orch.note)
        else:
            st.info(orch.note)
        return

    st.session_state.view = None

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
