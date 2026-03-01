from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from app.config import settings


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv", ".md"}


class DocumentProcessor:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def is_supported(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in SUPPORTED_EXTENSIONS

    def file_hash(self, file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def load(self, file_path: Path) -> List[Document]:
        suffix = file_path.suffix.lower()
        loader_map = {
            ".pdf": lambda p: PyPDFLoader(str(p)),
            ".txt": lambda p: TextLoader(str(p), encoding="utf-8"),
            ".docx": lambda p: Docx2txtLoader(str(p)),
            ".csv": lambda p: CSVLoader(str(p), encoding="utf-8"),
            ".md": lambda p: UnstructuredMarkdownLoader(str(p)),
        }
        if suffix not in loader_map:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
        loader = loader_map[suffix](file_path)
        return loader.load()

    def split(self, documents: List[Document]) -> List[Document]:
        chunks = self._splitter.split_documents(documents)
        return chunks[: settings.max_chunks_per_doc]

    def process(self, file_path: Path, doc_id: str) -> List[Document]:
        raw = self.load(file_path)
        chunks = self.split(raw)
        file_hash = self.file_hash(file_path)
        mime_type, _ = mimetypes.guess_type(str(file_path))
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "doc_id": doc_id,
                    "source_file": file_path.name,
                    "file_hash": file_hash,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "mime_type": mime_type or "application/octet-stream",
                }
            )
        return chunks

    def get_document_stats(self, file_path: Path) -> dict:
        raw = self.load(file_path)
        chunks = self.split(raw)
        total_chars = sum(len(d.page_content) for d in raw)
        return {
            "page_count": len(raw),
            "chunk_count": len(chunks),
            "total_characters": total_chars,
            "avg_chunk_size": total_chars // max(len(chunks), 1),
        }
