from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain.embeddings.base import Embeddings

from app.config import settings


def _build_huggingface_embeddings() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.hf_embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _build_ollama_embeddings() -> Embeddings:
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )


def _build_openai_embeddings() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.openai_embed_model,
        openai_api_key=settings.openai_api_key,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    provider = settings.embedding_provider
    builders = {
        "huggingface": _build_huggingface_embeddings,
        "ollama": _build_ollama_embeddings,
        "openai": _build_openai_embeddings,
    }
    if provider not in builders:
        raise ValueError(f"Unknown embedding provider: {provider}")
    return builders[provider]()


class EmbeddingsWrapper:
    def __init__(self):
        self._embeddings: Embeddings | None = None

    def _get(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._get().embed_query(text)

    def as_langchain(self) -> Embeddings:
        return self._get()
