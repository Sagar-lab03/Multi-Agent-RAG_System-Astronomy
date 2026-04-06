"""
Streamlit debug UI for RAG QA: retrieval → answer (Gemini) → optional verification.
Run from project root: streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

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

from rag_system.ingest.store import get_connection, init_schema
from rag_system.qa.answerer import AnswerConfig, answer_question
from rag_system.qa.verifier import verify_answer
from rag_system.retrieval import RetrieverConfig, retrieve_context


def _run_pipeline(
    db_path: Path,
    query: str,
    *,
    mode: str,
    top_k: int,
    alpha: float,
    run_verify: bool,
):
    conn = get_connection(db_path)
    init_schema(conn)
    cfg = RetrieverConfig(mode=mode, top_k=top_k, alpha=alpha)
    try:
        context = retrieve_context(conn, query, cfg)
    except Exception as e:
        conn.close()
        return None, None, None, f"Retrieval failed: {e}"
    conn.close()

    if not context:
        return None, None, None, "No context chunks retrieved. Run ingest + retrieval index first."

    try:
        acfg = AnswerConfig()
        result = answer_question(query, context, config=acfg)
    except Exception as e:
        return context, None, None, f"Answer generation failed: {e}"

    ver = None
    if run_verify:
        try:
            ver = verify_answer(query, result.answer, context)
        except Exception as e:
            st.warning(f"Verification failed: {e}")
    return context, result, ver, None


def main() -> None:
    st.set_page_config(page_title="RAG QA (debug)", layout="wide")
    st.title("Multi-Agent RAG — QA debug console")
    st.caption("DOCUMENT_SEARCH path: hybrid/semantic/lexical retrieval → Gemini answer → optional verifier.")

    with st.sidebar:
        st.header("Settings")
        db_default = str(_ROOT / "data" / "processed" / "chunks.db")
        db_path = st.text_input("SQLite chunk DB", value=db_default)
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

    with st.spinner("Retrieving context and generating answer…"):
        context, result, ver, err = _run_pipeline(
            path,
            q,
            mode=mode,
            top_k=int(top_k),
            alpha=float(alpha),
            run_verify=run_verify,
        )

    if err:
        st.error(err)
        if context is not None:
            with st.expander("Retrieved context (partial / before answer failed)", expanded=True):
                for idx, c in enumerate(context, start=1):
                    st.text((c.get("text") or "")[:1500])
                    st.divider()
        return

    st.subheader("Answer")
    st.markdown(result.answer)

    st.subheader("Sources")
    if not result.citations:
        st.info("No inline citations detected in the answer.")
    else:
        for c in result.citations:
            src = c.doc_source or ""
            page = f", page **{c.page}**" if c.page is not None else ""
            st.markdown(f"- **{c.label}** `chunk_id={c.chunk_id}` — `{src}`{page}")

    if ver is not None:
        st.subheader("Verification")
        col1, col2 = st.columns(2)
        col1.metric("Grounded", "yes" if ver.is_grounded else "no")
        col2.metric("Complete", "yes" if ver.is_complete else "no")
        if ver.issues:
            for i in ver.issues:
                labels = ", ".join(i.citation_labels) if i.citation_labels else "—"
                st.warning(f"**{i.type}** (citations: {labels})\n\n{i.description}")
        else:
            st.success("No issues reported.")

    with st.expander("Retrieved context (chunks)", expanded=False):
        for idx, c in enumerate(context or [], start=1):
            score = c.get("score", 0)
            src = c.get("doc_source", "")
            meta = c.get("metadata") or {}
            page = meta.get("page")
            st.markdown(f"**[{idx}]** score={score:.4f} · `{src}`" + (f" · page {page}" if page is not None else ""))
            st.text((c.get("text") or "")[:2000] + ("…" if len(c.get("text") or "") > 2000 else ""))
            st.divider()


if __name__ == "__main__":
    main()
