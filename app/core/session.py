from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from app.config import settings


class ChatMessage:
    def __init__(self, question: str, answer: str, sources: List[str], timestamp: str | None = None):
        self.question = question
        self.answer = answer
        self.sources = sources
        self.timestamp = timestamp or datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        return cls(
            question=data["question"],
            answer=data["answer"],
            sources=data.get("sources", []),
            timestamp=data.get("timestamp"),
        )


class Session:
    def __init__(self, session_id: str, name: str, doc_ids: List[str] | None = None):
        self.session_id = session_id
        self.name = name
        self.doc_ids = doc_ids or []
        self.messages: List[ChatMessage] = []
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at

    def add_message(self, message: ChatMessage) -> None:
        self.messages.append(message)
        self.updated_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "doc_ids": self.doc_ids,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        s = cls(
            session_id=data["session_id"],
            name=data["name"],
            doc_ids=data.get("doc_ids", []),
        )
        s.messages = [ChatMessage.from_dict(m) for m in data.get("messages", [])]
        s.created_at = data.get("created_at", datetime.utcnow().isoformat())
        s.updated_at = data.get("updated_at", s.created_at)
        return s


class SessionManager:
    def __init__(self):
        self._sessions_dir = settings.sessions_dir

    def _path(self, session_id: str) -> Path:
        return self._sessions_dir / f"{session_id}.json"

    def create_session(self, name: str, doc_ids: List[str] | None = None) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id, name=name, doc_ids=doc_ids)
        self._save(session)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        path = self._path(session_id)
        if not path.exists():
            return None
        with open(path) as f:
            return Session.from_dict(json.load(f))

    def _save(self, session: Session) -> None:
        with open(self._path(session.session_id), "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def add_message(self, session_id: str, message: ChatMessage) -> Optional[Session]:
        session = self.get_session(session_id)
        if session is None:
            return None
        session.add_message(message)
        self._save(session)
        return session

    def list_sessions(self) -> List[dict]:
        sessions = []
        for path in self._sessions_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                sessions.append(
                    {
                        "session_id": data["session_id"],
                        "name": data["name"],
                        "message_count": len(data.get("messages", [])),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)

    def delete_session(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear_old_sessions(self) -> int:
        cutoff = datetime.utcnow() - timedelta(hours=settings.session_ttl_hours)
        removed = 0
        for path in self._sessions_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                updated = datetime.fromisoformat(data.get("updated_at", "2000-01-01"))
                if updated < cutoff:
                    path.unlink()
                    removed += 1
            except Exception:
                continue
        return removed

    def update_session_docs(self, session_id: str, doc_ids: List[str]) -> Optional[Session]:
        session = self.get_session(session_id)
        if session is None:
            return None
        session.doc_ids = doc_ids
        session.updated_at = datetime.utcnow().isoformat()
        self._save(session)
        return session
