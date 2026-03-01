from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Analytics | RAG Studio", page_icon="📊", layout="wide")

st.title("Analytics & Insights")
st.caption("Monitor your knowledge base and query performance")


def get_health() -> dict | None:
    try:
        resp = requests.get(f"{API_BASE}/health/", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def get_documents() -> list:
    try:
        resp = requests.get(f"{API_BASE}/documents/", timeout=10)
        return resp.json() if resp.status_code == 200 else []
    except Exception:
        return []


def get_sessions() -> list:
    try:
        resp = requests.get(f"{API_BASE}/sessions/", timeout=10)
        return resp.json() if resp.status_code == 200 else []
    except Exception:
        return []


health = get_health()
documents = get_documents()
sessions = get_sessions()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Documents", len(documents))
col2.metric("Total Chunks", sum(d.get("chunk_count", 0) for d in documents))
col3.metric("Chat Sessions", len(sessions))
col4.metric("Total Messages", sum(s.get("message_count", 0) for s in sessions))

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Document Breakdown")
    if documents:
        for doc in documents:
            with st.expander(doc["filename"], expanded=False):
                st.markdown(f"**Document ID:** `{doc['doc_id']}`")
                st.markdown(f"**Chunks:** {doc['chunk_count']}")
                st.markdown(f"**Pages:** {doc['page_count']}")
                st.markdown(f"**Characters:** {doc['total_characters']:,}")
                size_kb = doc.get("file_size_bytes", 0) / 1024
                st.markdown(f"**File size:** {size_kb:.1f} KB")
                st.markdown(f"**Uploaded:** {doc['upload_time'][:10]}")
    else:
        st.info("No documents indexed yet.")

with right:
    st.subheader("Session Activity")
    if sessions:
        total_msgs = sum(s.get("message_count", 0) for s in sessions)
        avg_msgs = total_msgs / len(sessions) if sessions else 0
        st.metric("Average messages per session", f"{avg_msgs:.1f}")
        st.divider()
        for session in sessions[:10]:
            with st.expander(session["name"], expanded=False):
                st.markdown(f"**Session ID:** `{session['session_id'][:8]}...`")
                st.markdown(f"**Messages:** {session['message_count']}")
                st.markdown(f"**Created:** {session['created_at'][:10]}")
                st.markdown(f"**Last active:** {session['updated_at'][:10]}")
    else:
        st.info("No sessions yet.")

if health:
    st.divider()
    st.subheader("System Configuration")
    config_cols = st.columns(3)
    with config_cols[0]:
        st.markdown(f"**LLM Provider:** {health['llm_provider'].upper()}")
        st.markdown(f"**Embedding:** {health['embedding_provider'].upper()}")
    with config_cols[1]:
        st.markdown(f"**API Version:** {health['version']}")
        st.markdown(f"**API Status:** {'Online' if health['status'] == 'ok' else 'Offline'}")
    with config_cols[2]:
        if health.get("ollama_status"):
            ollama = health["ollama_status"]
            if ollama.get("status") == "ok":
                models = ", ".join(ollama.get("models", [])[:3]) or "none"
                st.markdown(f"**Ollama Models:** {models}")
            else:
                st.markdown(f"**Ollama:** Offline")
