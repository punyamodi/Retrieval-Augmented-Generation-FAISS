from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Settings | RAG Studio", page_icon="⚙️", layout="wide")

st.title("Settings & Configuration")
st.caption("Configure the LLM provider, embedding model, and retrieval parameters")

ENV_FILE = Path(".env")


def load_env() -> dict:
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def save_env(data: dict) -> None:
    lines = [f"{k}={v}" for k, v in data.items()]
    ENV_FILE.write_text("\n".join(lines) + "\n")


env = load_env()

tab_llm, tab_embed, tab_retrieval, tab_health = st.tabs(
    ["LLM Provider", "Embeddings", "Retrieval", "System Health"]
)

with tab_llm:
    st.subheader("Language Model Settings")

    llm_provider = st.selectbox(
        "LLM Provider",
        options=["ollama", "openai"],
        index=0 if env.get("LLM_PROVIDER", "ollama") == "ollama" else 1,
        help="Choose the LLM provider for answer generation",
    )
    env["LLM_PROVIDER"] = llm_provider

    if llm_provider == "ollama":
        ollama_url = st.text_input(
            "Ollama Base URL",
            value=env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        env["OLLAMA_BASE_URL"] = ollama_url

        ollama_model = st.text_input(
            "Ollama Model",
            value=env.get("OLLAMA_MODEL", "mistral:7b"),
            help="e.g. mistral:7b, llama3:8b, phi3:mini, gemma2:9b",
        )
        env["OLLAMA_MODEL"] = ollama_model

        st.markdown("**Popular Ollama models:**")
        model_table = {
            "mistral:7b": "Fast, good for RAG",
            "llama3:8b": "Meta Llama 3 8B",
            "phi3:mini": "Microsoft Phi-3 Mini (fast)",
            "gemma2:9b": "Google Gemma 2 9B",
            "dolphin-mistral:7b": "Uncensored Mistral",
        }
        st.table(model_table)

    elif llm_provider == "openai":
        openai_key = st.text_input(
            "OpenAI API Key",
            value=env.get("OPENAI_API_KEY", ""),
            type="password",
        )
        env["OPENAI_API_KEY"] = openai_key

        openai_model = st.selectbox(
            "OpenAI Model",
            options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"],
            index=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"].index(
                env.get("OPENAI_MODEL", "gpt-3.5-turbo")
            ),
        )
        env["OPENAI_MODEL"] = openai_model

with tab_embed:
    st.subheader("Embedding Model Settings")
    st.info("Changes to embeddings require re-indexing all documents.")

    embed_provider = st.selectbox(
        "Embedding Provider",
        options=["huggingface", "ollama", "openai"],
        index=["huggingface", "ollama", "openai"].index(
            env.get("EMBEDDING_PROVIDER", "huggingface")
        ),
    )
    env["EMBEDDING_PROVIDER"] = embed_provider

    if embed_provider == "huggingface":
        hf_model = st.text_input(
            "HuggingFace Model",
            value=env.get("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            help="Model ID from HuggingFace Hub",
        )
        env["HF_EMBED_MODEL"] = hf_model
        st.markdown("**Recommended models:**")
        st.table(
            {
                "all-MiniLM-L6-v2": "Fast, 384-dim, good general purpose",
                "all-mpnet-base-v2": "Better quality, 768-dim",
                "bge-large-en-v1.5": "State-of-the-art for retrieval",
            }
        )

    elif embed_provider == "ollama":
        ollama_embed = st.text_input(
            "Ollama Embed Model",
            value=env.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        )
        env["OLLAMA_EMBED_MODEL"] = ollama_embed

    elif embed_provider == "openai":
        openai_embed = st.selectbox(
            "OpenAI Embed Model",
            options=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            index=0,
        )
        env["OPENAI_EMBED_MODEL"] = openai_embed

with tab_retrieval:
    st.subheader("Retrieval & Chunking Parameters")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Document Chunking**")
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=200,
            max_value=4000,
            value=int(env.get("CHUNK_SIZE", 1000)),
            step=100,
        )
        env["CHUNK_SIZE"] = str(chunk_size)

        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=800,
            value=int(env.get("CHUNK_OVERLAP", 200)),
            step=50,
        )
        env["CHUNK_OVERLAP"] = str(chunk_overlap)

    with col2:
        st.markdown("**Retrieval Settings**")
        retrieval_k = st.slider(
            "Top-K Results",
            min_value=1,
            max_value=20,
            value=int(env.get("RETRIEVAL_K", 6)),
        )
        env["RETRIEVAL_K"] = str(retrieval_k)

        multi_query_count = st.slider(
            "Multi-Query Variants",
            min_value=2,
            max_value=10,
            value=int(env.get("MULTI_QUERY_COUNT", 5)),
        )
        env["MULTI_QUERY_COUNT"] = str(multi_query_count)

        compression_enabled = st.toggle(
            "Contextual Compression (default)",
            value=env.get("COMPRESSION_ENABLED", "true").lower() == "true",
        )
        env["COMPRESSION_ENABLED"] = str(compression_enabled).lower()

with tab_health:
    st.subheader("System Health")
    if st.button("Refresh Status", type="primary"):
        try:
            resp = requests.get(f"{API_BASE}/health/", timeout=5)
            if resp.status_code == 200:
                health = resp.json()
                st.success("API server is running")
                st.json(health)
            else:
                st.error(f"API returned {resp.status_code}")
        except Exception as exc:
            st.error(f"Cannot reach API: {exc}")
    else:
        st.info("Click Refresh Status to check system health")

    st.divider()
    st.markdown("**Start API server:**")
    st.code("uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload", language="bash")
    st.markdown("**Start UI:**")
    st.code("streamlit run ui/Home.py --server.port 8501", language="bash")

st.divider()
if st.button("Save Settings", type="primary", use_container_width=True):
    save_env(env)
    st.success("Settings saved to .env file. Restart the API server for changes to take effect.")
