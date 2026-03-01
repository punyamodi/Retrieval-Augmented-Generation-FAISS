from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.core.document_processor import DocumentProcessor
from app.core.session import SessionManager
from app.core.vectorstore import VectorStoreManager


_vector_store_manager: VectorStoreManager | None = None
_session_manager: SessionManager | None = None
_document_processor: DocumentProcessor | None = None


def get_vector_store_manager() -> VectorStoreManager:
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_document_processor() -> DocumentProcessor:
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor


VSManagerDep = Annotated[VectorStoreManager, Depends(get_vector_store_manager)]
SessionManagerDep = Annotated[SessionManager, Depends(get_session_manager)]
DocProcessorDep = Annotated[DocumentProcessor, Depends(get_document_processor)]
