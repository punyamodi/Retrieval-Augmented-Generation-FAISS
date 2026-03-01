from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Chat | RAG Studio", page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        word-wrap: break-word;
    }
    .assistant-bubble {
        background: #f0f2f6;
        color: #1a1a2e;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 85%;
        word-wrap: break-word;
    }
    .source-chip {
        display: inline-block;
        background: #e8ecf8;
        color: #4a5568;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 0.1rem;
    }
    .response-time { font-size: 0.7rem; color: #aaa; margin-top: 0.3rem; }
    .chat-container { max-height: 65vh; overflow-y: auto; padding: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def create_session(name: str, doc_ids: list) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}/sessions/",
            json={"name": name, "doc_ids": doc_ids},
            timeout=10,
        )
        return resp.json() if resp.status_code == 201 else None
    except Exception:
        return None


def delete_session(session_id: str) -> bool:
    try:
        resp = requests.delete(f"{API_BASE}/sessions/{session_id}", timeout=10)
        return resp.status_code == 204
    except Exception:
        return False


def query(question: str, session_id: str, doc_ids: list, use_multi_query: bool, use_compression: bool) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}/query/",
            json={
                "question": question,
                "session_id": session_id,
                "doc_ids": doc_ids if doc_ids else None,
                "use_multi_query": use_multi_query,
                "use_compression": use_compression,
            },
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Query failed ({resp.status_code}): {resp.json().get('detail', resp.text)}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server.")
    except Exception as exc:
        st.error(f"Error: {exc}")
    return None


st.title("Chat with Documents")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.markdown("### Session Settings")

    documents = get_documents()
    doc_options = {d["filename"]: d["doc_id"] for d in documents}

    if not doc_options:
        st.warning("No documents indexed. Upload documents first.")
        selected_docs = []
    else:
        selected_filenames = st.multiselect(
            "Filter by documents (leave empty to search all)",
            options=list(doc_options.keys()),
            default=[],
        )
        selected_docs = [doc_options[f] for f in selected_filenames]

    st.markdown("### Retrieval Settings")
    use_multi_query = st.toggle("Multi-Query Expansion", value=True, help="Generate multiple query variants for better recall")
    use_compression = st.toggle("Contextual Compression", value=True, help="Filter irrelevant content from retrieved chunks")

    st.divider()
    st.markdown("### Session")

    sessions = get_sessions()
    session_names = {s["session_id"]: s["name"] for s in sessions}

    if sessions:
        selected_session = st.selectbox(
            "Load existing session",
            options=["-- New Session --"] + list(session_names.keys()),
            format_func=lambda x: "-- New Session --" if x == "-- New Session --" else session_names.get(x, x),
        )
        if selected_session != "-- New Session --" and selected_session != st.session_state.session_id:
            try:
                resp = requests.get(f"{API_BASE}/sessions/{selected_session}", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.session_id = selected_session
                    st.session_state.chat_history = [
                        {"role": "user", "content": m["question"], "sources": m["sources"], "time_ms": 0}
                        for m in data["messages"]
                    ] + [
                        {"role": "assistant", "content": m["answer"], "sources": m["sources"], "time_ms": 0}
                        for m in data["messages"]
                    ]
                    hist = []
                    for m in data["messages"]:
                        hist.append({"role": "user", "content": m["question"], "sources": [], "time_ms": 0})
                        hist.append({"role": "assistant", "content": m["answer"], "sources": m["sources"], "time_ms": 0})
                    st.session_state.chat_history = hist
            except Exception:
                pass

    new_session_name = st.text_input("New session name", value="My Session")
    if st.button("Start New Session", type="primary", use_container_width=True):
        result = create_session(new_session_name, selected_docs)
        if result:
            st.session_state.session_id = result["session_id"]
            st.session_state.chat_history = []
            st.success(f"Session started: {result['name']}")

    if st.session_state.session_id:
        st.caption(f"Active: `{st.session_state.session_id[:8]}...`")
        col_clear, col_del = st.columns(2)
        with col_clear:
            if st.button("Clear History", use_container_width=True):
                try:
                    requests.delete(f"{API_BASE}/sessions/{st.session_state.session_id}/history", timeout=10)
                    st.session_state.chat_history = []
                    st.rerun()
                except Exception:
                    pass
        with col_del:
            if st.button("End Session", use_container_width=True, type="secondary"):
                delete_session(st.session_state.session_id)
                st.session_state.session_id = None
                st.session_state.chat_history = []
                st.rerun()


main_col, _ = st.columns([4, 0])

with main_col:
    if not st.session_state.session_id:
        st.info("Start a session using the sidebar to begin chatting.")
    else:
        st.markdown("#### Conversation")
        chat_container = st.container()

        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="user-bubble">{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    sources_html = ""
                    for src in msg.get("sources", []):
                        if isinstance(src, dict):
                            label = src.get("source_file", "source")
                        else:
                            label = str(src)
                        sources_html += f'<span class="source-chip">{label}</span>'

                    time_html = ""
                    if msg.get("time_ms"):
                        time_html = f'<div class="response-time">{msg["time_ms"]}ms</div>'

                    st.markdown(
                        f'<div class="assistant-bubble">{msg["content"]}'
                        f'<br/><small>{sources_html}</small>'
                        f"{time_html}</div>",
                        unsafe_allow_html=True,
                    )

        st.divider()
        with st.form("chat_form", clear_on_submit=True):
            question = st.text_area(
                "Ask a question",
                placeholder="What are the key findings in these documents?",
                height=80,
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send", type="primary", use_container_width=True)

        if submitted and question.strip():
            st.session_state.chat_history.append(
                {"role": "user", "content": question, "sources": [], "time_ms": 0}
            )
            with st.spinner("Retrieving and generating answer..."):
                result = query(question, st.session_state.session_id, selected_docs, use_multi_query, use_compression)
            if result:
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "time_ms": result.get("response_time_ms", 0),
                    }
                )
            st.rerun()
