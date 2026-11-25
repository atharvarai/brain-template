"""Audit log: who did what through the brain, kept apart from traces.

Traces are for debugging and may be sampled or pruned; the audit log is the record of
actions and refusals: every tool call, every call the policy blocked, every approval
decision and every rejected token.
"""

import json
import sqlite3
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from core.access.types import User


class AuditAction(StrEnum):
    TOOL_CALL = "tool_call"  # a tool ran (outcome ok or error)
    TOOL_DENIED = "tool_denied"  # the policy blocked a call
    APPROVAL = "approval"  # a person approved or declined an action
    AUTH_FAILED = "auth_failed"  # a request's token was rejected
    GUARD_BLOCKED = "guard_blocked"  # a guardrail refused a message or replaced an answer


@dataclass(frozen=True, slots=True)
class AuditEntry:
    action: AuditAction
    user_id: str
    roles: tuple[str, ...]
    target: str  # tool name, run id, or route
    outcome: str
    trace_id: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def of(
        cls, action: AuditAction, user: User, target: str, outcome: str, **kw: Any
    ) -> "AuditEntry":
        return cls(action, user.id, tuple(sorted(user.roles)), target, outcome, **kw)


class AuditLog(Protocol):
    def record(self, entry: AuditEntry) -> None: ...

    def recent(self, limit: int) -> list[AuditEntry]: ...


class InMemoryAuditLog:
    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []

    def record(self, entry: AuditEntry) -> None:
        self.entries.append(entry)

    def recent(self, limit: int) -> list[AuditEntry]:
        return list(reversed(self.entries))[:limit]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    at       TEXT NOT NULL,
    action   TEXT NOT NULL,
    user_id  TEXT NOT NULL,
    roles    TEXT NOT NULL,
    target   TEXT NOT NULL,
    outcome  TEXT NOT NULL,
    trace_id TEXT,
    detail   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_by_user ON audit (user_id, at);
"""


class SqliteAuditLog:
    """Append-only: the brain never updates or deletes entries."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(_SCHEMA)

    def record(self, entry: AuditEntry) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO audit (at, action, user_id, roles, target, outcome, trace_id, detail)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.at.isoformat(), entry.action.value, entry.user_id,
                    json.dumps(entry.roles), entry.target, entry.outcome, entry.trace_id,
                    json.dumps(dict(entry.detail), ensure_ascii=False, default=str),
                ),
            )  # fmt: skip

    def recent(self, limit: int) -> list[AuditEntry]:
        with self._lock:
            rows = self._db.execute(
                "SELECT at, action, user_id, roles, target, outcome, trace_id, detail"
                " FROM audit ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            AuditEntry(
                AuditAction(action), user_id, tuple(json.loads(roles)), target, outcome,
                trace_id, json.loads(detail), datetime.fromisoformat(at),
            )
            for at, action, user_id, roles, target, outcome, trace_id, detail in rows
        ]  # fmt: skip


class RedactingAuditLog:
    """Decorator: cleans personal data out of entry details before the inner log stores
    them. Tool arguments often carry it (an email to send to, a phone number to call)."""

    def __init__(self, inner: AuditLog, redact: Callable[[str], str]) -> None:
        self._inner = inner
        self._redact = redact

    def record(self, entry: AuditEntry) -> None:
        self._inner.record(replace(entry, detail=self._clean(dict(entry.detail))))

    def recent(self, limit: int) -> list[AuditEntry]:
        return self._inner.recent(limit)

    def _clean(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, dict):
            return {key: self._clean(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [self._clean(item) for item in value]
        return value
