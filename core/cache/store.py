"""Where cached answers live. SQLite with brute-force similarity: fine for thousands of
entries per scope; a vector database or Redis adapter can replace it behind the protocol."""

import json
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np

Vector = np.ndarray


@dataclass(frozen=True, slots=True)
class CachedAnswer:
    question: str
    body: dict[str, Any]  # what the pipeline needs to rebuild the answer
    created_at: datetime


class CacheStore(Protocol):
    def get(self, key: str) -> CachedAnswer | None: ...

    def nearest(
        self, scope: str, vector: Vector, figures: frozenset[str]
    ) -> tuple[CachedAnswer, float] | None:
        """The most similar unexpired entry in ``scope`` with exactly these figures."""
        ...

    def put(
        self,
        key: str,
        scope: str,
        question: str,
        figures: frozenset[str],
        vector: Vector | None,
        body: dict[str, Any],
        ttl: timedelta,
    ) -> None: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    key        TEXT PRIMARY KEY,
    scope      TEXT NOT NULL,
    question   TEXT NOT NULL,
    figures    TEXT NOT NULL,
    vector     BLOB,
    body       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    hits       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS answers_by_scope ON answers (scope, expires_at);
"""


class SqliteCacheStore:
    def __init__(self, path: Path, now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._now = now
        with self._lock:
            self._db.executescript(_SCHEMA)

    def get(self, key: str) -> CachedAnswer | None:
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT question, body, created_at FROM answers WHERE key = ? AND expires_at > ?",
                (key, self._now().isoformat()),
            ).fetchone()
            if row:
                self._db.execute("UPDATE answers SET hits = hits + 1 WHERE key = ?", (key,))
        return _entry(row) if row else None

    def nearest(
        self, scope: str, vector: Vector, figures: frozenset[str]
    ) -> tuple[CachedAnswer, float] | None:
        with self._lock:
            rows = self._db.execute(
                "SELECT key, question, body, created_at, vector FROM answers"
                " WHERE scope = ? AND expires_at > ? AND vector IS NOT NULL AND figures = ?",
                (scope, self._now().isoformat(), json.dumps(sorted(figures))),
            ).fetchall()
        if not rows:
            return None
        matrix = np.vstack([np.frombuffer(row[4], dtype=np.float32) for row in rows])
        scores = matrix @ np.asarray(vector, dtype=np.float32)  # unit vectors: cosine
        best = int(np.argmax(scores))
        key, question, body, created, _ = rows[best]
        with self._lock, self._db:
            self._db.execute("UPDATE answers SET hits = hits + 1 WHERE key = ?", (key,))
        return _entry((question, body, created)), float(scores[best])

    def put(
        self,
        key: str,
        scope: str,
        question: str,
        figures: frozenset[str],
        vector: Vector | None,
        body: dict[str, Any],
        ttl: timedelta,
    ) -> None:
        now = self._now()
        blob = None if vector is None else np.asarray(vector, dtype=np.float32).tobytes()
        with self._lock, self._db:
            self._db.execute("DELETE FROM answers WHERE expires_at <= ?", (now.isoformat(),))
            self._db.execute(
                "INSERT OR REPLACE INTO answers"
                " (key, scope, question, figures, vector, body, created_at, expires_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (key, scope, question, json.dumps(sorted(figures)), blob, json.dumps(body),
                 now.isoformat(), (now + ttl).isoformat()),
            )  # fmt: skip

    def size(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM answers").fetchone()[0])


def _entry(row: tuple[str, str, str]) -> CachedAnswer:
    question, body, created = row
    return CachedAnswer(question, json.loads(body), datetime.fromisoformat(created))
