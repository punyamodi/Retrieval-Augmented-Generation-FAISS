from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from app.core.session import ChatMessage, Session, SessionManager


class TestChatMessage:
    def test_to_dict(self):
        msg = ChatMessage(question="What is RAG?", answer="RAG is...", sources=["doc1.pdf"])
        d = msg.to_dict()
        assert d["question"] == "What is RAG?"
        assert d["answer"] == "RAG is..."
        assert d["sources"] == ["doc1.pdf"]
        assert "timestamp" in d

    def test_from_dict_roundtrip(self):
        msg = ChatMessage(question="Q", answer="A", sources=["s1"])
        restored = ChatMessage.from_dict(msg.to_dict())
        assert restored.question == msg.question
        assert restored.answer == msg.answer
        assert restored.sources == msg.sources


class TestSession:
    def test_create_session(self):
        s = Session(session_id="abc", name="test", doc_ids=["d1"])
        assert s.session_id == "abc"
        assert s.name == "test"
        assert s.doc_ids == ["d1"]
        assert len(s.messages) == 0

    def test_add_message_updates_timestamp(self):
        s = Session(session_id="abc", name="test")
        old_ts = s.updated_at
        import time; time.sleep(0.01)
        s.add_message(ChatMessage("q", "a", []))
        assert s.updated_at >= old_ts
        assert len(s.messages) == 1

    def test_to_dict_from_dict_roundtrip(self):
        s = Session(session_id="abc", name="Test", doc_ids=["d1", "d2"])
        s.add_message(ChatMessage("q1", "a1", ["src1"]))
        d = s.to_dict()
        s2 = Session.from_dict(d)
        assert s2.session_id == s.session_id
        assert s2.name == s.name
        assert len(s2.messages) == 1


class TestSessionManager:
    def setup_method(self):
        import tempfile
        self._tmp = tempfile.mkdtemp()
        from unittest.mock import patch
        from app import config
        self._patcher = patch.object(config.settings, "sessions_dir", Path(self._tmp))
        self._patcher.start()
        self.manager = SessionManager()
        self.manager._sessions_dir = Path(self._tmp)

    def teardown_method(self):
        self._patcher.stop()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_create_and_get_session(self):
        session = self.manager.create_session("My Session", doc_ids=["d1"])
        assert session.session_id is not None
        fetched = self.manager.get_session(session.session_id)
        assert fetched is not None
        assert fetched.name == "My Session"
        assert fetched.doc_ids == ["d1"]

    def test_get_nonexistent_session(self):
        assert self.manager.get_session("nonexistent-id") is None

    def test_add_message_to_session(self):
        session = self.manager.create_session("test")
        msg = ChatMessage("hello", "world", ["src"])
        updated = self.manager.add_message(session.session_id, msg)
        assert updated is not None
        assert len(updated.messages) == 1
        persisted = self.manager.get_session(session.session_id)
        assert len(persisted.messages) == 1

    def test_list_sessions(self):
        self.manager.create_session("s1")
        self.manager.create_session("s2")
        sessions = self.manager.list_sessions()
        assert len(sessions) == 2

    def test_delete_session(self):
        session = self.manager.create_session("to-delete")
        assert self.manager.delete_session(session.session_id)
        assert self.manager.get_session(session.session_id) is None

    def test_delete_nonexistent_returns_false(self):
        assert not self.manager.delete_session("no-such-id")

    def test_update_session_docs(self):
        session = self.manager.create_session("test", doc_ids=["d1"])
        updated = self.manager.update_session_docs(session.session_id, ["d1", "d2", "d3"])
        assert updated.doc_ids == ["d1", "d2", "d3"]
