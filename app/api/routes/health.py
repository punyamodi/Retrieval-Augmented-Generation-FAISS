from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import get_session_manager, get_vector_store_manager
from app.api.models import HealthResponse
from app.config import settings
from app.core.llm import check_ollama_health

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthResponse)
def health_check():
    vs_manager = get_vector_store_manager()
    session_manager = get_session_manager()

    ollama_status = None
    if settings.llm_provider == "ollama":
        ollama_status = check_ollama_health()

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        document_count=len(vs_manager.list_documents()),
        session_count=len(session_manager.list_sessions()),
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        ollama_status=ollama_status,
    )
