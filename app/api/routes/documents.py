from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.api.dependencies import DocProcessorDep, VSManagerDep
from app.api.models import DocumentInfo, DocumentUploadResponse, ErrorResponse
from app.config import settings
from app.core.document_processor import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/documents", tags=["documents"])


def _save_upload(file: UploadFile, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with open(dest, "wb") as f:
        while chunk := file.file.read(8192):
            size += len(chunk)
            if size > settings.max_upload_size_mb * 1024 * 1024:
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds {settings.max_upload_size_mb} MB limit",
                )
            f.write(chunk)
    return size


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    vs_manager: VSManagerDep,
    doc_processor: DocProcessorDep,
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_path = Path(file.filename)
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_path.suffix}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    doc_id = str(uuid.uuid4())[:8]
    dest = settings.documents_dir / f"{doc_id}_{file.filename}"
    file_size = _save_upload(file, dest)

    try:
        stats = doc_processor.get_document_stats(dest)
        chunks = doc_processor.process(dest, doc_id)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Failed to process document: {exc}")

    doc_meta = {
        "filename": file.filename,
        "chunk_count": len(chunks),
        "page_count": stats["page_count"],
        "total_characters": stats["total_characters"],
        "upload_time": datetime.utcnow().isoformat(),
        "file_size_bytes": file_size,
        "stored_path": str(dest),
    }

    try:
        vs_manager.add_documents(doc_id, chunks, doc_meta)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to index document: {exc}")

    return DocumentUploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        chunk_count=len(chunks),
        page_count=stats["page_count"],
        total_characters=stats["total_characters"],
    )


@router.get("/", response_model=list[DocumentInfo])
def list_documents(vs_manager: VSManagerDep):
    return [DocumentInfo(**doc) for doc in vs_manager.list_documents()]


@router.get("/{doc_id}", response_model=DocumentInfo)
def get_document(doc_id: str, vs_manager: VSManagerDep):
    meta = vs_manager.get_document_meta(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return DocumentInfo(doc_id=doc_id, **meta)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, vs_manager: VSManagerDep):
    meta = vs_manager.get_document_meta(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    stored_path = meta.get("stored_path")
    if stored_path:
        Path(stored_path).unlink(missing_ok=True)

    deleted = vs_manager.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
