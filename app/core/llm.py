from __future__ import annotations

from functools import lru_cache

from langchain.llms.base import BaseLLM
from langchain.chat_models.base import BaseChatModel

from app.config import settings


def _build_ollama_llm() -> BaseLLM:
    from langchain_ollama import OllamaLLM

    return OllamaLLM(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
    )


def _build_openai_llm() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.1,
    )


@lru_cache(maxsize=1)
def get_llm():
    provider = settings.llm_provider
    if provider == "ollama":
        return _build_ollama_llm()
    if provider == "openai":
        return _build_openai_llm()
    raise ValueError(f"Unknown LLM provider: {provider}")


def check_ollama_health() -> dict:
    import httpx

    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"status": "ok", "models": models}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
