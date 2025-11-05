from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, TypedDict

from ..utils.logger import get_logger


MessageRole = Literal["user", "assistant", "system", "tool"]


class ConversationMessage(TypedDict):
    role: MessageRole
    content: str
    timestamp: str


class ConversationStore:
    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self.storage_path.write_text(json.dumps({}, indent=2), encoding="utf-8")
        self._lock = threading.Lock()
        self.logger = get_logger(self.__class__.__name__)

    def _read(self) -> Dict[str, List[ConversationMessage]]:
        try:
            raw = self.storage_path.read_text(encoding="utf-8")
            return json.loads(raw)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _write(self, data: Dict[str, List[ConversationMessage]]) -> None:
        serialized = json.dumps(data, indent=2)
        self.storage_path.write_text(serialized, encoding="utf-8")

    def append_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
    ) -> None:
        message: ConversationMessage = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            data = self._read()
            messages = data.setdefault(session_id, [])
            messages.append(message)
            data[session_id] = messages
            self._write(data)
        self.logger.debug("Appended %s message to session %s", role, session_id)

    def get_history(self, session_id: str) -> List[ConversationMessage]:
        with self._lock:
            data = self._read()
            return data.get(session_id, []).copy()

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            data = self._read()
            if session_id in data:
                del data[session_id]
                self._write(data)
        self.logger.info("Reset conversation session %s", session_id)
