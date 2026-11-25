"""Paused runs survive a round trip through SQLite, expire, and are claimed once."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from core.agent import (
    InMemoryRunStore,
    PausedRun,
    RunClosedError,
    RunNotFoundError,
    RunProgress,
    SqliteRunStore,
    ToolActivity,
)
from core.config import ModelSpec, Provider
from core.llm import Message, Role, StopReason, ToolCall, ToolResult
from core.rag import Chunk

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)
CALL = ToolCall("c1", "cancel_order", {"id": 7, "reason": "late"})


def paused(run_id: str = "r1", ttl: timedelta = timedelta(hours=1)) -> PausedRun:
    progress = RunProgress(
        messages=[
            Message(Role.USER, "cancel 7"),
            Message(Role.ASSISTANT, "", tool_calls=(CALL, ToolCall("c2", "find_order", {}))),
        ],
        steps=1,
        input_tokens=10,
        output_tokens=2,
        latency_ms=40,
        tool_calls=2,
        texts=["Let me check."],
        activity=[ToolActivity("find_order", True, 5, 1)],
        spec=ModelSpec(provider=Provider.FAKE, model="m"),
        stop_reason=StopReason.TOOL_USE,
        pending=(CALL,),
        held={"c2": ToolResult("c2", "find_order", "found", False)},
        batch=("c1", "c2"),
    )
    chunk = Chunk("orders.md", 0, "Orders", "Cancelling", "You can cancel.", frozenset({"x"}))
    return PausedRun(run_id, NOW, NOW + ttl, progress, (chunk,))


@pytest.fixture(params=["sqlite", "memory"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> SqliteRunStore | InMemoryRunStore:
    clock = lambda: NOW  # noqa: E731
    if request.param == "sqlite":
        return SqliteRunStore(tmp_path / "runs.sqlite3", now=clock)
    return InMemoryRunStore(now=clock)


def test_a_paused_run_comes_back_exactly(store: SqliteRunStore | InMemoryRunStore) -> None:
    original = paused()
    store.save(original)

    loaded = store.get("r1")

    assert loaded.progress == original.progress
    assert loaded.chunks == original.chunks
    assert loaded.expires_at == original.expires_at


def test_a_run_is_claimed_once(store: SqliteRunStore | InMemoryRunStore) -> None:
    store.save(paused())

    store.claim("r1")

    with pytest.raises(RunClosedError):
        store.claim("r1")
    with pytest.raises(RunClosedError):
        store.get("r1")


def test_unknown_and_expired_runs_are_not_found(
    store: SqliteRunStore | InMemoryRunStore,
) -> None:
    store.save(paused("old", ttl=timedelta(seconds=0)))

    with pytest.raises(RunNotFoundError):
        store.get("nope")
    with pytest.raises(RunNotFoundError):
        store.get("old")
    with pytest.raises((RunClosedError, RunNotFoundError)):
        store.claim("old")
