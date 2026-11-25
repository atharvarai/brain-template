"""Runs paused for approval: stored, looked up, and claimed exactly once.

Claiming is atomic, so the same approval can never be applied twice, even when two requests
race. A paused run expires after the pack's ``approval_ttl_minutes``.
"""

import json
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from core.agent.codec import chunk_from_dict, chunk_to_dict, progress_from_dict, progress_to_dict
from core.agent.errors import RunClosedError, RunNotFoundError
from core.agent.types import RunProgress
from core.rag import Chunk

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class PausedRun:
    run_id: str
    created_at: datetime
    expires_at: datetime
    progress: RunProgress
    chunks: tuple[Chunk, ...]  # the documents the run was grounded in, for its sources


class RunStore(Protocol):
    def save(self, run: PausedRun) -> None: ...

    def get(self, run_id: str) -> PausedRun:
        """The run, if it is still waiting. Raises ``RunNotFoundError`` or ``RunClosedError``."""
        ...

    def claim(self, run_id: str) -> None:
        """Mark the run as resumed. Raises if it is gone, expired or already claimed."""
        ...


class InMemoryRunStore:
    """Paused runs in a dict: for tests and single-process demos."""

    def __init__(self, now: Clock = _utc_now) -> None:
        self._runs: dict[str, PausedRun] = {}
        self._claimed: set[str] = set()
        self._now = now

    def save(self, run: PausedRun) -> None:
        self._runs[run.run_id] = run

    def get(self, run_id: str) -> PausedRun:
        if run_id in self._claimed:
            raise RunClosedError(f"run {run_id} was already resumed")
        run = self._runs.get(run_id)
        if run is None or run.expires_at <= self._now():
            raise RunNotFoundError(f"no paused run {run_id}, or it expired")
        return run

    def claim(self, run_id: str) -> None:
        self.get(run_id)
        self._claimed.add(run_id)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    claimed    INTEGER NOT NULL DEFAULT 0,
    body       TEXT NOT NULL
);
"""


class SqliteRunStore:
    def __init__(self, path: Path, now: Clock = _utc_now) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._now = now
        with self._lock:
            self._db.executescript(_SCHEMA)

    def save(self, run: PausedRun) -> None:
        body = {
            "progress": progress_to_dict(run.progress),
            "chunks": [chunk_to_dict(c) for c in run.chunks],
        }
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO runs (run_id, created_at, expires_at, body) VALUES (?, ?, ?, ?)",
                (run.run_id, run.created_at.isoformat(), run.expires_at.isoformat(),
                 json.dumps(body)),
            )  # fmt: skip

    def get(self, run_id: str) -> PausedRun:
        with self._lock:
            row = self._db.execute(
                "SELECT created_at, expires_at, claimed, body FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise RunNotFoundError(f"no paused run {run_id}")
        created, expires, claimed, body = row
        if claimed:
            raise RunClosedError(f"run {run_id} was already resumed")
        expires_at = datetime.fromisoformat(expires)
        if expires_at <= self._now():
            raise RunNotFoundError(f"run {run_id} expired")
        data = json.loads(body)
        return PausedRun(
            run_id,
            datetime.fromisoformat(created),
            expires_at,
            progress_from_dict(data["progress"]),
            tuple(chunk_from_dict(c) for c in data["chunks"]),
        )

    def claim(self, run_id: str) -> None:
        with self._lock, self._db:
            cursor = self._db.execute(
                "UPDATE runs SET claimed = 1 WHERE run_id = ? AND claimed = 0 AND expires_at > ?",
                (run_id, self._now().isoformat()),
            )
        if cursor.rowcount != 1:
            raise RunClosedError(f"run {run_id} was already resumed or has expired")

    def close(self) -> None:
        with self._lock:
            self._db.close()
