from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    page_count: int
    total_characters: int
    upload_time: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    page_count: int
    total_characters: int
    upload_time: str
    file_size_bytes: int = 0


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None
    use_multi_query: bool = True
    use_compression: bool = True
    top_k: int = Field(default=6, ge=1, le=20)


class SourceDocument(BaseModel):
    content: str
    source_file: str
    chunk_index: int
    doc_id: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    question: str
    session_id: Optional[str] = None
    response_time_ms: int = 0


class SessionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    doc_ids: Optional[List[str]] = None


class SessionInfo(BaseModel):
    session_id: str
    name: str
    message_count: int
    created_at: str
    updated_at: str


class ChatMessageResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    timestamp: str


class SessionDetail(BaseModel):
    session_id: str
    name: str
    doc_ids: List[str]
    messages: List[ChatMessageResponse]
    created_at: str
    updated_at: str


class HealthResponse(BaseModel):
    status: str
    version: str
    document_count: int
    session_count: int
    llm_provider: str
    embedding_provider: str
    ollama_status: Optional[dict] = None


class ErrorResponse(BaseModel):
    detail: str
    error_type: str = "error"
