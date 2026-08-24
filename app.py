"""Streamlit chat UI for the Knowledge Assistant.

Run:
    streamlit run app.py

Sidebar lets you switch role (department access), backend, and profile
(baseline vs improved) live, so the same UI can demo both the failure modes
and the fixes side by side.
"""
from __future__ import annotations

import streamlit as st

from src.access_control import get_allowed_departments
from src.config import PROFILES, ROLE_DEPARTMENT_ACCESS, Settings
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Northwind Knowledge Assistant", page_icon="📚", layout="wide")


def esc(text: str) -> str:
    """Escape '$' so Streamlit's markdown renderer doesn't interpret dollar
    amounts (e.g. "$100/person") as LaTeX math delimiters."""
    return text.replace("$", "\\$")


@st.cache_resource(show_spinner="Loading index...")
def load_pipeline(backend: str, profile_name: str) -> RAGPipeline:
    settings = Settings()
    settings.backend = backend
    settings.profile_name = profile_name
    return RAGPipeline(settings)


with st.sidebar:
    st.title("📚 Northwind Knowledge Assistant")
    st.caption("RAG over Finance / HR / IT / Legal / Sales policy documents")

    role = st.selectbox("Signed in as (role)", options=sorted(ROLE_DEPARTMENT_ACCESS.keys()), index=sorted(ROLE_DEPARTMENT_ACCESS.keys()).index("admin"))
    st.caption(f"Access: {', '.join(get_allowed_departments(role)) or '(no departments — none)'}")

    backend = st.selectbox("Backend", options=["local", "github", "azure"], index=0, help="local = fully offline (hashed embeddings, extractive answers). github = free real LLM+embeddings via GitHub Models, local hybrid search. azure = Azure OpenAI + Azure AI Search (production).")
    profile_name = st.selectbox("Pipeline profile", options=list(PROFILES.keys()), index=list(PROFILES.keys()).index("improved"), help="baseline reproduces the assignment's naive-RAG failure modes; improved applies all fixes.")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Backend must already be ingested: `python scripts/ingest.py --profile <name> --backend <name>`")

try:
    pipeline = load_pipeline(backend, profile_name)
except Exception as e:  # noqa: BLE001
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(esc(msg["content"]))
        if msg.get("debug"):
            with st.expander("Retrieval & debug details"):
                st.json(msg["debug"])

query = st.chat_input("Ask about Finance, HR, IT, Legal, or Sales policies…")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(esc(query))

    allowed = get_allowed_departments(role)
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
    result = pipeline.ask(query, allowed, history)

    with st.chat_message("assistant"):
        if result.ambiguous:
            text = "That's ambiguous — could you clarify which of these you mean?\n\n" + "\n".join(
                f"- {t}" for t in result.clarification_options
            )
            st.markdown(esc(text))
            debug = {"rewritten_query": result.rewritten_query, "ambiguous": True, "candidates": result.clarification_options}
        else:
            answer = result.answer
            st.markdown(esc(answer.text))
            if answer.citations:
                with st.expander(f"📎 {len(answer.citations)} citation(s)"):
                    for c in answer.citations:
                        eff = f" (effective {c.effective_date}, v{c.version})" if c.effective_date else f" (v{c.version})"
                        st.markdown(esc(f"**[{c.index}]** {c.title} — {c.section}{eff}  \n`{c.source_path}`"))
            if answer.guardrail and answer.guardrail.flagged:
                st.warning(f"⚠️ Guardrail flag: {answer.guardrail.flag_reason}")
            text = answer.text
            debug = {
                "rewritten_query": result.rewritten_query,
                "confidence": round(answer.confidence, 3),
                "insufficient_evidence": answer.insufficient_evidence,
                "used_llm": answer.used_llm,
                "groundedness": answer.guardrail.groundedness_ratio if answer.guardrail else None,
                "timings_ms": {k: round(v, 2) for k, v in result.timings_ms.items()},
                "retrieved_chunks": [
                    {"doc_id": h.record["doc_id"], "section": h.record["section"], "department": h.record["department"], "score": round(h.score, 4)}
                    for h in result.hits
                ],
            }
        with st.expander("Retrieval & debug details"):
            st.json(debug)

    st.session_state.messages.append({"role": "assistant", "content": text, "debug": debug})
