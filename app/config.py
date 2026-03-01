from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAG Studio"
    app_version: str = "2.0.0"
    debug: bool = False

    storage_dir: Path = Path("storage")
    documents_dir: Path = Path("storage/documents")
    indexes_dir: Path = Path("storage/indexes")
    sessions_dir: Path = Path("storage/sessions")

    llm_provider: Literal["ollama", "openai"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:7b"
    ollama_embed_model: str = "nomic-embed-text"

    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    openai_embed_model: str = "text-embedding-3-small"

    embedding_provider: Literal["ollama", "openai", "huggingface"] = "huggingface"
    hf_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_doc: int = 500

    retrieval_k: int = 6
    multi_query_count: int = 5
    compression_enabled: bool = True
    similarity_threshold: float = 0.3

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    max_upload_size_mb: int = 50
    session_ttl_hours: int = 24

    def model_post_init(self, __context: object) -> None:
        for d in (self.documents_dir, self.indexes_dir, self.sessions_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
