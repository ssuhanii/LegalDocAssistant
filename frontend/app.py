"""
frontend/app.py
Streamlit UI for LegalDocAssistant.

Run with:
    streamlit run frontend/app.py
"""

import sys
from pathlib import Path

import httpx
import streamlit as st

# ── Make sure project root is on the path ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LegalDocAssistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* Main background */
        .stApp { background: #0f1117; color: #e2e8f0; }

        /* Header banner */
        .hero {
            /* background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%); */
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-bottom: 2rem;
            /* border: 1px solid #2d5a8e44; */
        }
        .hero h1 { font-size: 2.2rem; font-weight: 700; color: #eba817; margin: 0; }
        .hero p  { color: #ffffff; margin: 0.4rem 0 0; font-size: 1.05rem; }

        /* Answer card */
        .answer-card {
            background: #1a2235;
            border: 1px solid #2d5a8e55;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            line-height: 1.75;
        }

        /* Source card */
        .source-card {
            background: #151c2c;
            border-left: 3px solid #3b82f6;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin: 0.6rem 0;
            font-size: 0.88rem;
            color: #94a3b8;
        }
        .source-card strong { color: #60a5fa; }

        /* Status badges */
        .badge-ok  { background:#14532d; color:#86efac; padding:3px 10px; border-radius:20px; font-size:0.8rem; }
        .badge-err { background:#7f1d1d; color:#fca5a5; padding:3px 10px; border-radius:20px; font-size:0.8rem; }

        /* Input overrides */
        textarea, input { background:#1a2235 !important; color:#e2e8f0 !important; }
        .stButton>button {
            background: linear-gradient(135deg, #1d4ed8, #2563eb);
            color: white; border: none; border-radius: 8px;
            padding: 0.55rem 1.6rem; font-weight: 600;
            transition: opacity .2s;
        }
        .stButton>button:hover { opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helper: call API ──────────────────────────────────────────────────────────
def api_get(path: str) -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, payload: dict) -> dict | None:
    try:
        r = httpx.post(f"{API_BASE}{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e))
        st.error(f"API error {e.response.status_code}: {detail}")
        return None
    except Exception as e:
        st.error(f"Connection error: {e}. Is the FastAPI server running?")
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ LegalDocAssistant")
    st.markdown("---")

    # Health check
    health = api_get("/health")
    if health:
        ready = health.get("index_ready", False)
        badge = '<span class="badge-ok">● Index ready</span>' if ready else '<span class="badge-err">● No index</span>'
        st.markdown(f"**API status:** {badge}", unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-err">● API offline</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Documents list
    st.markdown("### 📁 Loaded Documents")
    sources = api_get("/sources")
    if sources and sources["count"] > 0:
        for doc in sources["documents"]:
            st.markdown(f"- 📄 `{doc}`")
    else:
        st.info("No documents indexed yet.")

    st.markdown("---")

    # Ingest button
    st.markdown("### Re-index Documents")
    st.caption("Drop new PDFs into `data/documents/` then click below.")
    if st.button("▶ Run Ingestion"):
        with st.spinner("Indexing… this may take a few minutes."):
            resp = api_post("/ingest", {})
            if resp:
                st.success(resp.get("message", "Done!"))

    st.markdown("---")

    # Advanced settings
    with st.expander("⚙️ Advanced settings"):
        retrieve_k = st.slider("Retrieve candidates (k)", 5, 20, 10)
        rerank_n   = st.slider("Rerank top-n chunks",     1, 10,  3)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>⚖️ Legal Document Assistant - Harvey</h1>
        <p>Ask Harvey questions about your legal documents. Powered by local AI, no data leaves your machine.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander("Source passages"):
                for src in msg["sources"]:
                    score_text = (
                        f" &nbsp;|&nbsp; score: <strong>{src['rerank_score']:.3f}</strong>"
                        if src.get("rerank_score") is not None else ""
                    )
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>{src["source"]}</strong> — page {src["page"]}{score_text}<br>'
                        f'<em>{src["snippet"]}</em>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Input
question = st.chat_input("Ask a legal question, e.g. 'What are the indemnification clauses?'")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "raw_content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Build history from all PREVIOUS turns (exclude the one just added)
    history_for_api = [
        {"role": m["role"], "content": m["raw_content"]}
        for m in st.session_state.messages[:-1]   # everything before the current question
        if m["role"] in ("user", "assistant") and m.get("raw_content")
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = api_post(
                "/query",
                {
                    "question":     question,
                    "retrieve_k":   retrieve_k,
                    "rerank_n":     rerank_n,
                    "chat_history": history_for_api,
                },
            )

        if result:
            answer  = result["answer"]
            sources = result.get("sources", [])
            latency = result.get("latency", {})   # ← new

            st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

    # ── Latency breakdown ─────────────────────────────────────────────────
            if latency:
                with st.expander("⚡ Response latency"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Retrieval", f"{latency.get('retrieval_s', 0)}s")
                    col2.metric("Reranking", f"{latency.get('rerank_s', 0)}s")
                    col3.metric("LLM",       f"{latency.get('llm_s', 0)}s")
                    col4.metric("Total",      f"{latency.get('total_s', 0)}s")

    # ── Source passages ───────────────────────────────────────────────────
            if sources:
                with st.expander("Source passages"):
                    for src in sources:
                        score_text = (
                            f" &nbsp;|&nbsp; score: <strong>{src['rerank_score']:.3f}</strong>"
                            if src.get("rerank_score") is not None else ""
                )
                st.markdown(
                    f'<div class="source-card">'
                    f'<strong>{src["source"]}</strong> — page {src["page"]}{score_text}<br>'
                    f'<em>{src["snippet"]}</em>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.session_state.messages.append({
                "role":        "assistant",
                "content":     f'<div class="answer-card">{answer}</div>',
                "raw_content": answer,
                "sources":     sources,
                "latency":     latency,   # ← store for chat history render too
    })