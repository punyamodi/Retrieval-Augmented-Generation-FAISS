from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="RAG Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.25rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-ok { background: #d4edda; color: #155724; }
    .status-error { background: #f8d7da; color: #721c24; }
    .feature-card {
        border: 1px solid #e8ecef;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: box-shadow 0.2s;
    }
    .feature-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.1); }
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_health() -> dict | None:
    try:
        resp = requests.get(f"{API_BASE}/health/", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


st.markdown('<div class="main-header">RAG Studio</div>', unsafe_allow_html=True)
st.markdown("**Intelligent document Q&A powered by FAISS vector search and LLMs**")
st.divider()

health = fetch_health()

if health:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{health["document_count"]}</div>'
            f'<div class="metric-label">Documents Indexed</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{health["session_count"]}</div>'
            f'<div class="metric-label">Chat Sessions</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        llm = health["llm_provider"].upper()
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{llm}</div>'
            f'<div class="metric-label">LLM Provider</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        embed = health["embedding_provider"].upper()
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{embed}</div>'
            f'<div class="metric-label">Embedding Model</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    ollama = health.get("ollama_status")
    if ollama:
        if ollama.get("status") == "ok":
            models = ", ".join(ollama.get("models", [])[:4]) or "none loaded"
            st.success(f"Ollama connected | Available models: {models}")
        else:
            st.warning(f"Ollama not reachable: {ollama.get('error', 'unknown error')}")
else:
    st.error("Cannot connect to the RAG Studio API server. Make sure it is running on port 8000.")
    st.code("uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload", language="bash")

st.divider()

st.subheader("How it works")

fc1, fc2, fc3 = st.columns(3)
with fc1:
    st.markdown(
        """<div class="feature-card">
        <h4>1. Upload Documents</h4>
        <p>Upload PDFs, Word documents, text files, CSVs, or Markdown files. The system automatically chunks and indexes them into a FAISS vector store.</p>
        </div>""",
        unsafe_allow_html=True,
    )
with fc2:
    st.markdown(
        """<div class="feature-card">
        <h4>2. Ask Questions</h4>
        <p>Ask natural language questions. The multi-query retriever generates multiple query variants to maximize recall from your document corpus.</p>
        </div>""",
        unsafe_allow_html=True,
    )
with fc3:
    st.markdown(
        """<div class="feature-card">
        <h4>3. Get Answers with Sources</h4>
        <p>Receive grounded answers with source citations. Contextual compression filters noise and surfaces the most relevant passages.</p>
        </div>""",
        unsafe_allow_html=True,
    )

st.divider()

st.subheader("Architecture")
st.markdown(
    """
```
User Query
    │
    ▼
Multi-Query Expansion (LLM generates 5 variants)
    │
    ▼
FAISS Vector Search (cosine similarity over all variants)
    │
    ▼
Contextual Compression (LLM filters irrelevant content)
    │
    ▼
RAG Chain (LLM generates grounded answer with citations)
    │
    ▼
Response + Source Documents
```
"""
)

st.divider()
st.caption(f"RAG Studio v{health['version'] if health else '2.0.0'} | Navigate using the sidebar")
