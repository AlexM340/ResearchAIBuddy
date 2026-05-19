"""
Persistent chat session manager for CerebrumAI.
Stores chat sessions in JSON files under data/chat_sessions.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


class ChatSessionManager:
    """Manage chat sessions with file-based persistence."""

    def __init__(self, base_path: str = "./data/chat_sessions"):
        self.base_path = Path(base_path)
        self.sessions_path = self.base_path / "sessions"
        self.index_file = self.base_path / "sessions_index.json"

        self.base_path.mkdir(parents=True, exist_ok=True)
        self.sessions_path.mkdir(parents=True, exist_ok=True)

        self.index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "sessions" in data:
                        return data
            except Exception as exc:
                logger.error("Failed to load chat sessions index: %s", exc)

        return {"version": "1.0", "sessions": {}}

    def _save_index(self) -> None:
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error("Failed to save chat sessions index: %s", exc)

    def _session_file(self, session_id: str) -> Path:
        safe_id = "".join(ch for ch in session_id if ch.isalnum() or ch in {"_", "-"})
        return self.sessions_path / f"{safe_id}.json"

    def _save_session_data(self, session_data: Dict[str, Any]) -> None:
        try:
            session_file = self._session_file(session_data["id"])
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error("Failed to save chat session %s: %s", session_data.get("id"), exc)

    def _session_summary(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        messages = session_data.get("messages", [])
        return {
            "id": session_data["id"],
            "title": session_data.get("title", "Chat nou"),
            "topic_collection": session_data.get("topic_collection", ""),
            "query_mode": session_data.get("query_mode", "topic_general"),
            "created_at": session_data.get("created_at", _now_iso()),
            "updated_at": session_data.get("updated_at", _now_iso()),
            "message_count": len(messages),
            "migrated_to_db": bool(session_data.get("migrated_to_db", False)),
            "db_session_id": session_data.get("db_session_id", ""),
        }

    def _upsert_index_entry(self, session_data: Dict[str, Any]) -> None:
        self.index["sessions"][session_data["id"]] = self._session_summary(session_data)
        self._save_index()

    def create_session(
        self,
        title: str = "Chat nou",
        topic_collection: str = "",
        query_mode: str = "topic_general",
    ) -> Dict[str, Any]:
        session_id = f"chat_{uuid.uuid4().hex[:12]}"
        now = _now_iso()
        session_data = {
            "id": session_id,
            "title": (title or "Chat nou").strip() or "Chat nou",
            "topic_collection": (topic_collection or "").strip(),
            "query_mode": query_mode or "topic_general",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }

        self._save_session_data(session_data)
        self._upsert_index_entry(session_data)
        return self.index["sessions"][session_id]

    def list_sessions(
        self,
        topic_collection: Optional[str] = None,
        include_migrated: bool = True,
    ) -> List[Dict[str, Any]]:
        sessions = list(self.index.get("sessions", {}).values())
        if topic_collection is not None:
            sessions = [
                item for item in sessions if item.get("topic_collection", "") == topic_collection
            ]
        if not include_migrated:
            sessions = [item for item in sessions if not item.get("migrated_to_db", False)]
        sessions.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return sessions

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None

        session_file = self._session_file(session_id)
        if session_file.exists():
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as exc:
                logger.error("Failed to load chat session %s: %s", session_id, exc)
        return None

    def save_session(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        title: Optional[str] = None,
        topic_collection: Optional[str] = None,
        query_mode: Optional[str] = None,
    ) -> bool:
        session_data = self.load_session(session_id)
        if not session_data:
            return False

        session_data["messages"] = messages or []
        session_data["updated_at"] = _now_iso()

        if title is not None:
            cleaned_title = title.strip() or "Chat nou"
            session_data["title"] = cleaned_title

        if topic_collection is not None:
            session_data["topic_collection"] = topic_collection.strip()

        if query_mode is not None:
            session_data["query_mode"] = query_mode

        self._save_session_data(session_data)
        self._upsert_index_entry(session_data)
        return True

    def append_exchange(self, session_id: str, exchange: Dict[str, Any]) -> bool:
        session_data = self.load_session(session_id)
        if not session_data:
            return False

        messages = session_data.get("messages", [])
        messages.append(exchange)
        session_data["messages"] = messages
        session_data["updated_at"] = _now_iso()

        # Auto-title first exchange if still default.
        current_title = (session_data.get("title") or "").strip().lower()
        if current_title in {"", "chat nou"}:
            question = (exchange.get("question") or "").strip()
            if question:
                session_data["title"] = (question[:60] + "...") if len(question) > 60 else question

        self._save_session_data(session_data)
        self._upsert_index_entry(session_data)
        return True

    def clear_session(self, session_id: str) -> bool:
        session_data = self.load_session(session_id)
        if not session_data:
            return False

        session_data["messages"] = []
        session_data["updated_at"] = _now_iso()
        self._save_session_data(session_data)
        self._upsert_index_entry(session_data)
        return True

    def rename_session(self, session_id: str, new_title: str) -> bool:
        session_data = self.load_session(session_id)
        if not session_data:
            return False

        session_data["title"] = (new_title or "").strip() or "Chat nou"
        session_data["updated_at"] = _now_iso()
        self._save_session_data(session_data)
        self._upsert_index_entry(session_data)
        return True

    def delete_session(self, session_id: str) -> bool:
        removed_anything = False

        if session_id in self.index.get("sessions", {}):
            del self.index["sessions"][session_id]
            self._save_index()
            removed_anything = True

        session_file = self._session_file(session_id)
        if session_file.exists():
            try:
                session_file.unlink()
                removed_anything = True
            except Exception as exc:
                logger.error("Failed to delete chat session file %s: %s", session_id, exc)

        return removed_anything

    def mark_session_migrated(self, session_id: str, db_session_id: str = "") -> bool:
        session_data = self.load_session(session_id)
        if not session_data:
            return False

        session_data["migrated_to_db"] = True
        if db_session_id:
            session_data["db_session_id"] = str(db_session_id)
        session_data["updated_at"] = _now_iso()
        self._save_session_data(session_data)
        self._upsert_index_entry(session_data)
        return True
