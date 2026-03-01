from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_storage(tmp_path: Path, monkeypatch):
    from app import config

    monkeypatch.setattr(config.settings, "documents_dir", tmp_path / "documents")
    monkeypatch.setattr(config.settings, "indexes_dir", tmp_path / "indexes")
    monkeypatch.setattr(config.settings, "sessions_dir", tmp_path / "sessions")

    for d in (config.settings.documents_dir, config.settings.indexes_dir, config.settings.sessions_dir):
        d.mkdir(parents=True, exist_ok=True)

    return tmp_path


@pytest.fixture
def sample_txt_file(tmp_path: Path) -> Path:
    content = "\n\n".join(
        [
            "Artificial intelligence is transforming industries worldwide.",
            "Machine learning models can now process natural language with remarkable accuracy.",
            "Retrieval-Augmented Generation combines retrieval systems with generative models.",
            "FAISS is a library for efficient similarity search and clustering of dense vectors.",
            "Vector databases store embeddings and enable semantic search at scale.",
        ]
    )
    file = tmp_path / "sample.txt"
    file.write_text(content, encoding="utf-8")
    return file


@pytest.fixture
def sample_pdf_file(tmp_path: Path) -> Path:
    try:
        from reportlab.pdfgen import canvas

        pdf_path = tmp_path / "sample.pdf"
        c = canvas.Canvas(str(pdf_path))
        c.drawString(100, 750, "RAG Studio Test Document")
        c.drawString(100, 720, "This document is used for testing the document processing pipeline.")
        c.save()
        return pdf_path
    except ImportError:
        pytest.skip("reportlab not installed, skipping PDF test")
