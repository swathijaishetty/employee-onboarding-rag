"""Thread-safe, bounded storage for independent browser conversations."""

from collections import deque
from dataclasses import dataclass, field
from threading import Lock, RLock
import time

from .config_v3 import SETTINGS, Settings
from .memory_v3 import SessionMemory


@dataclass
class StoredSession:
    memory: SessionMemory
    last_access: float
    lock: Lock = field(default_factory=Lock)


class SessionStore:
    def __init__(self, settings: Settings = SETTINGS):
        self.settings = settings
        self._sessions: dict[str, StoredSession] = {}
        self._lock = RLock()

    def get(self, session_id: str) -> StoredSession:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            session = self._sessions.get(session_id)
            if session is None:
                if len(self._sessions) >= self.settings.max_sessions:
                    oldest = min(self._sessions, key=lambda key: self._sessions[key].last_access)
                    self._sessions.pop(oldest)
                session = StoredSession(SessionMemory(self.settings), now)
                self._sessions[session_id] = session
            session.last_access = now
            return session

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    @property
    def count(self) -> int:
        with self._lock:
            self._prune(time.monotonic())
            return len(self._sessions)

    def _prune(self, now: float) -> None:
        expired = [
            key for key, session in self._sessions.items()
            if now - session.last_access > self.settings.session_ttl_seconds
        ]
        for key in expired:
            self._sessions.pop(key, None)


class RateLimiter:
    def __init__(self, settings: Settings = SETTINGS):
        self.limit = settings.requests_per_minute
        self._requests: dict[str, deque[float]] = {}
        self._lock = RLock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - 60
        with self._lock:
            timestamps = self._requests.setdefault(key, deque())
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()
            if len(timestamps) >= self.limit:
                return False
            timestamps.append(now)
            return True
