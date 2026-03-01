from __future__ import annotations

import time
from typing import List

from fastapi import APIRouter, HTTPException

from app.api.dependencies import SessionManagerDep, VSManagerDep
from app.api.models import QueryRequest, QueryResponse, SourceDocument
from app.core.chain import build_rag_chain
from app.core.session import ChatMessage

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/", response_model=QueryResponse)
def query_knowledge_base(
    request: QueryRequest,
    vs_manager: VSManagerDep,
    session_manager: SessionManagerDep,
):
    store = vs_manager.get_merged_store(request.doc_ids)
    if store is None:
        raise HTTPException(
            status_code=404,
            detail="No documents indexed. Upload documents first.",
        )

    session = None
    history: List[dict] = []
    if request.session_id:
        session = session_manager.get_session(request.session_id)
        if session:
            history = [m.to_dict() for m in session.messages[-6:]]

    chain = build_rag_chain(store, use_multi_query=request.use_multi_query)

    t0 = time.time()
    try:
        result = chain.run_with_history(request.question, history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")
    elapsed_ms = int((time.time() - t0) * 1000)

    sources = [
        SourceDocument(
            content=doc.page_content[:500],
            source_file=doc.metadata.get("source_file", "unknown"),
            chunk_index=doc.metadata.get("chunk_index", 0),
            doc_id=doc.metadata.get("doc_id", ""),
        )
        for doc in result["source_documents"]
    ]

    if session and request.session_id:
        msg = ChatMessage(
            question=request.question,
            answer=result["answer"],
            sources=[s.source_file for s in sources],
        )
        session_manager.add_message(request.session_id, msg)

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        question=request.question,
        session_id=request.session_id,
        response_time_ms=elapsed_ms,
    )


@router.post("/similarity", response_model=List[SourceDocument])
def similarity_search(
    request: QueryRequest,
    vs_manager: VSManagerDep,
):
    results = vs_manager.similarity_search_with_score(
        query=request.question,
        k=request.top_k,
        doc_ids=request.doc_ids,
    )
    if not results:
        return []
    return [
        SourceDocument(
            content=doc.page_content[:500],
            source_file=doc.metadata.get("source_file", "unknown"),
            chunk_index=doc.metadata.get("chunk_index", 0),
            doc_id=doc.metadata.get("doc_id", ""),
            score=round(float(score), 4),
        )
        for doc, score in results
    ]
