from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app import config

    monkeypatch.setattr(config.settings, "documents_dir", tmp_path / "documents")
    monkeypatch.setattr(config.settings, "indexes_dir", tmp_path / "indexes")
    monkeypatch.setattr(config.settings, "sessions_dir", tmp_path / "sessions")
    for d in (config.settings.documents_dir, config.settings.indexes_dir, config.settings.sessions_dir):
        d.mkdir(parents=True, exist_ok=True)

    from app.api import dependencies
    dependencies._vector_store_manager = None
    dependencies._session_manager = None
    dependencies._document_processor = None

    return TestClient(app)


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/api/v1/health/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "document_count" in body
        assert "session_count" in body


class TestDocumentsEndpoint:
    def test_list_documents_empty(self, client):
        resp = client.get("/api/v1/documents/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_upload_unsupported_type(self, client):
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.exe", b"binary", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_get_nonexistent_document(self, client):
        resp = client.get("/api/v1/documents/nonexistent")
        assert resp.status_code == 404

    def test_delete_nonexistent_document(self, client):
        resp = client.delete("/api/v1/documents/nonexistent")
        assert resp.status_code == 404


class TestSessionsEndpoint:
    def test_list_sessions_empty(self, client):
        resp = client.get("/api/v1/sessions/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_session(self, client):
        resp = client.post(
            "/api/v1/sessions/",
            json={"name": "Test Session", "doc_ids": []},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "Test Session"
        assert "session_id" in body

    def test_get_session(self, client):
        create_resp = client.post("/api/v1/sessions/", json={"name": "S1"})
        sid = create_resp.json()["session_id"]
        get_resp = client.get(f"/api/v1/sessions/{sid}")
        assert get_resp.status_code == 200
        assert get_resp.json()["session_id"] == sid

    def test_delete_session(self, client):
        create_resp = client.post("/api/v1/sessions/", json={"name": "To Delete"})
        sid = create_resp.json()["session_id"]
        del_resp = client.delete(f"/api/v1/sessions/{sid}")
        assert del_resp.status_code == 204
        get_resp = client.get(f"/api/v1/sessions/{sid}")
        assert get_resp.status_code == 404

    def test_get_nonexistent_session(self, client):
        resp = client.get("/api/v1/sessions/no-such-id")
        assert resp.status_code == 404

    def test_clear_session_history(self, client):
        create_resp = client.post("/api/v1/sessions/", json={"name": "History Test"})
        sid = create_resp.json()["session_id"]
        clear_resp = client.delete(f"/api/v1/sessions/{sid}/history")
        assert clear_resp.status_code == 200
        assert clear_resp.json()["messages"] == []


class TestQueryEndpoint:
    def test_query_with_no_documents(self, client):
        resp = client.post(
            "/api/v1/query/",
            json={"question": "What is RAG?"},
        )
        assert resp.status_code == 404
        assert "No documents" in resp.json()["detail"]
