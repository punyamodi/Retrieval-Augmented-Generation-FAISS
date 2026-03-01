from __future__ import annotations

from pathlib import Path

import pytest

from app.core.document_processor import DocumentProcessor, SUPPORTED_EXTENSIONS


class TestDocumentProcessor:
    def setup_method(self):
        self.processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)

    def test_supported_extensions(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".csv" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS

    def test_is_supported(self, sample_txt_file: Path):
        assert self.processor.is_supported(sample_txt_file)
        assert not self.processor.is_supported(Path("file.xyz"))

    def test_load_txt(self, sample_txt_file: Path):
        docs = self.processor.load(sample_txt_file)
        assert len(docs) > 0
        assert all(d.page_content for d in docs)

    def test_split_produces_chunks(self, sample_txt_file: Path):
        docs = self.processor.load(sample_txt_file)
        chunks = self.processor.split(docs)
        assert len(chunks) >= len(docs)

    def test_process_adds_metadata(self, sample_txt_file: Path):
        chunks = self.processor.process(sample_txt_file, doc_id="test123")
        assert all(c.metadata.get("doc_id") == "test123" for c in chunks)
        assert all(c.metadata.get("source_file") == sample_txt_file.name for c in chunks)
        assert all(isinstance(c.metadata.get("chunk_index"), int) for c in chunks)

    def test_file_hash_is_deterministic(self, sample_txt_file: Path):
        h1 = self.processor.file_hash(sample_txt_file)
        h2 = self.processor.file_hash(sample_txt_file)
        assert h1 == h2
        assert len(h1) == 16

    def test_unsupported_file_raises(self, tmp_path: Path):
        bad_file = tmp_path / "file.exe"
        bad_file.write_bytes(b"binary")
        with pytest.raises(ValueError, match="Unsupported file type"):
            self.processor.load(bad_file)

    def test_get_document_stats(self, sample_txt_file: Path):
        stats = self.processor.get_document_stats(sample_txt_file)
        assert "page_count" in stats
        assert "chunk_count" in stats
        assert "total_characters" in stats
        assert stats["total_characters"] > 0

    def test_chunk_metadata_indices(self, sample_txt_file: Path):
        chunks = self.processor.process(sample_txt_file, doc_id="abc")
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(chunks)
