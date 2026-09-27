from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock

from .config import settings


class MemoryService:
    def __init__(self, max_turns: int = settings.memory_max_turns) -> None:
        self.max_turns = max_turns
        self._sessions: dict[str, deque[dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=self.max_turns * 2)
        )
        self._lock = Lock()

    def get_messages(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._sessions.get(session_id, ()))

    def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        with self._lock:
            session = self._sessions[session_id]
            session.append({"role": "user", "content": user_message})
            session.append({"role": "assistant", "content": assistant_message})

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

