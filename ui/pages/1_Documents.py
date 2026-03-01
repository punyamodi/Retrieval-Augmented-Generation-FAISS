from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Documents | RAG Studio", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .doc-card {
        border: 1px solid #e0e4e8;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        background: #fafbfc;
    }
    .doc-title { font-weight: 600; font-size: 1rem; color: #1a1a2e; }
    .doc-meta { font-size: 0.8rem; color: #666; margin-top: 0.25rem; }
    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Document Management")
st.caption("Upload, view, and manage documents in the knowledge base")


def upload_file(file_bytes, filename: str) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}/documents/upload",
            files={"file": (filename, file_bytes, "application/octet-stream")},
            timeout=120,
        )
        if resp.status_code == 201:
            return resp.json()
        st.error(f"Upload failed ({resp.status_code}): {resp.json().get('detail', resp.text)}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server. Make sure it is running.")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
    return None


def get_documents() -> list:
    try:
        resp = requests.get(f"{API_BASE}/documents/", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def delete_document(doc_id: str) -> bool:
    try:
        resp = requests.delete(f"{API_BASE}/documents/{doc_id}", timeout=10)
        return resp.status_code == 204
    except Exception:
        return False


tab_upload, tab_library = st.tabs(["Upload Documents", "Document Library"])

with tab_upload:
    st.markdown("### Upload New Document")
    st.markdown(
        '<div class="upload-zone"><b>Supported formats:</b> PDF, TXT, DOCX, CSV, Markdown</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "docx", "csv", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("Index All Files", type="primary", use_container_width=True):
            progress = st.progress(0)
            results = []
            for i, f in enumerate(uploaded_files):
                with st.spinner(f"Indexing {f.name}..."):
                    result = upload_file(f.read(), f.name)
                    if result:
                        results.append(result)
                progress.progress((i + 1) / len(uploaded_files))

            progress.empty()
            if results:
                st.success(f"Successfully indexed {len(results)} document(s)")
                for r in results:
                    st.markdown(
                        f'<div class="doc-card">'
                        f'<div class="doc-title">{r["filename"]}</div>'
                        f'<div class="doc-meta">ID: {r["doc_id"]} | '
                        f'{r["chunk_count"]} chunks | '
                        f'{r["page_count"]} pages | '
                        f'{r["total_characters"]:,} characters</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

with tab_library:
    st.markdown("### Indexed Documents")

    col_refresh, col_stats = st.columns([1, 4])
    with col_refresh:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    documents = get_documents()

    if not documents:
        st.info("No documents indexed yet. Upload documents using the Upload tab.")
    else:
        with col_stats:
            total_chunks = sum(d.get("chunk_count", 0) for d in documents)
            total_chars = sum(d.get("total_characters", 0) for d in documents)
            st.markdown(
                f"**{len(documents)} documents** | {total_chunks:,} total chunks | {total_chars:,} total characters"
            )

        for doc in documents:
            col_info, col_action = st.columns([5, 1])
            with col_info:
                size_kb = doc.get("file_size_bytes", 0) / 1024
                st.markdown(
                    f'<div class="doc-card">'
                    f'<div class="doc-title">{doc["filename"]}</div>'
                    f'<div class="doc-meta">'
                    f'ID: <code>{doc["doc_id"]}</code> | '
                    f'{doc["chunk_count"]} chunks | '
                    f'{doc["page_count"]} pages | '
                    f'{size_kb:.1f} KB | '
                    f'Uploaded: {doc["upload_time"][:10]}'
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            with col_action:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Delete", key=f"del_{doc['doc_id']}", type="secondary"):
                    if delete_document(doc["doc_id"]):
                        st.success(f"Deleted {doc['filename']}")
                        st.rerun()
                    else:
                        st.error("Delete failed")
