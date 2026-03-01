from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import SessionManagerDep, VSManagerDep
from app.api.models import (
    ChatMessageResponse,
    SessionCreateRequest,
    SessionDetail,
    SessionInfo,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("/", response_model=SessionDetail, status_code=status.HTTP_201_CREATED)
def create_session(
    request: SessionCreateRequest,
    session_manager: SessionManagerDep,
    vs_manager: VSManagerDep,
):
    if request.doc_ids:
        for doc_id in request.doc_ids:
            if not vs_manager.document_exists(doc_id):
                raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    session = session_manager.create_session(name=request.name, doc_ids=request.doc_ids)
    return _to_detail(session)


@router.get("/", response_model=list[SessionInfo])
def list_sessions(session_manager: SessionManagerDep):
    return [SessionInfo(**s) for s in session_manager.list_sessions()]


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: str, session_manager: SessionManagerDep):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return _to_detail(session)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str, session_manager: SessionManagerDep):
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


@router.delete("/{session_id}/history", response_model=SessionDetail)
def clear_session_history(session_id: str, session_manager: SessionManagerDep):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    session.messages.clear()
    session_manager._save(session)
    return _to_detail(session)


@router.put("/{session_id}/documents", response_model=SessionDetail)
def update_session_documents(
    session_id: str,
    doc_ids: list[str],
    session_manager: SessionManagerDep,
    vs_manager: VSManagerDep,
):
    for doc_id in doc_ids:
        if not vs_manager.document_exists(doc_id):
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    session = session_manager.update_session_docs(session_id, doc_ids)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return _to_detail(session)


def _to_detail(session) -> SessionDetail:
    return SessionDetail(
        session_id=session.session_id,
        name=session.name,
        doc_ids=session.doc_ids,
        messages=[
            ChatMessageResponse(
                question=m.question,
                answer=m.answer,
                sources=m.sources,
                timestamp=m.timestamp,
            )
            for m in session.messages
        ],
        created_at=session.created_at,
        updated_at=session.updated_at,
    )
