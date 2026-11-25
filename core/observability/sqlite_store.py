"""SQLite trace store: zero-install default. Swap for a hosted backend behind the same ports."""

import json
import sqlite3
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from core.observability.serialize import trace_from_dict, trace_to_dict
from core.observability.types import Feedback, Rating, Trace

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id   TEXT PRIMARY KEY,
    operation  TEXT NOT NULL,
    started_at TEXT NOT NULL,
    status     TEXT NOT NULL,
    body       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS traces_by_time ON traces (started_at);
CREATE TABLE IF NOT EXISTS feedback (
    trace_id   TEXT PRIMARY KEY REFERENCES traces (trace_id),
    rating     TEXT NOT NULL,
    comment    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class SqliteTraceStore:
    """Implements ``TraceSink``, ``TraceReader`` and ``FeedbackSink``."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared across threads, guarded by a lock: writes are tiny.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(_SCHEMA)

    def record(self, trace: Trace) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO traces VALUES (?, ?, ?, ?, ?)",
                (
                    trace.trace_id,
                    trace.operation,
                    trace.started_at.isoformat(),
                    trace.status.value,
                    json.dumps(trace_to_dict(trace)),
                ),
            )

    def recent(self, limit: int, operation: str | None = None) -> list[Trace]:
        query = "SELECT body FROM traces"
        params: tuple[object, ...] = ()
        if operation:
            query += " WHERE operation = ?"
            params = (operation,)
        query += " ORDER BY started_at DESC LIMIT ?"
        with self._lock:
            rows = self._db.execute(query, (*params, limit)).fetchall()
        return [trace_from_dict(json.loads(body)) for (body,) in rows]

    def get(self, trace_id: str) -> Trace | None:
        with self._lock:
            row = self._db.execute(
                "SELECT body FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        return trace_from_dict(json.loads(row[0])) if row else None

    def add_feedback(self, feedback: Feedback) -> bool:
        created = feedback.created_at or datetime.now(UTC)
        with self._lock, self._db:
            exists = self._db.execute(
                "SELECT 1 FROM traces WHERE trace_id = ?", (feedback.trace_id,)
            ).fetchone()
            if not exists:
                return False
            self._db.execute(
                "INSERT OR REPLACE INTO feedback VALUES (?, ?, ?, ?)",
                (
                    feedback.trace_id,
                    feedback.rating.value,
                    feedback.comment,
                    created.isoformat(),
                ),
            )
        return True

    def feedback_for(self, trace_ids: Sequence[str]) -> dict[str, Feedback]:
        if not trace_ids:
            return {}
        marks = ",".join("?" * len(trace_ids))
        with self._lock:
            rows = self._db.execute(
                f"SELECT trace_id, rating, comment, created_at FROM feedback "
                f"WHERE trace_id IN ({marks})",
                tuple(trace_ids),
            ).fetchall()
        return {
            tid: Feedback(tid, Rating(rating), comment, datetime.fromisoformat(created))
            for tid, rating, comment, created in rows
        }

    def close(self) -> None:
        with self._lock:
            self._db.close()
