"""SQLite trace store, feedback and summaries."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.observability import (
    Event,
    Feedback,
    Rating,
    Span,
    SqliteTraceStore,
    Trace,
    TraceStatus,
    percentile,
    summarize,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def trace(
    trace_id: str,
    minutes: int = 0,
    operation: str = "chat",
    status: TraceStatus = TraceStatus.OK,
    ms: int = 100,
    cost: float | None = 0.001,
) -> Trace:
    return Trace(
        trace_id=trace_id,
        operation=operation,
        started_at=T0 + timedelta(minutes=minutes),
        duration_ms=ms,
        status=status,
        provider="fake",
        model="m1",
        input_tokens=10,
        output_tokens=5,
        cost_usd=cost,
        prompts=("system@v2#abc",),
        retrieved=("a.md#0",),
        cited=("a.md#0",),
        spans=(Span("retrieve", 0, 12, {"hits": 1}),),
        events=(Event("first_token", 40),),
        attributes={"pack": "Acme"},
    )


def test_traces_round_trip_and_come_back_newest_first(tmp_path: Path) -> None:
    store = SqliteTraceStore(tmp_path / "t.sqlite3")
    store.record(trace("old", minutes=0))
    store.record(trace("new", minutes=5))
    store.record(trace("judged", minutes=9, operation="judge"))

    assert store.get("old") == trace("old", minutes=0)
    assert store.get("missing") is None
    assert [t.trace_id for t in store.recent(10)] == ["judged", "new", "old"]
    assert [t.trace_id for t in store.recent(10, operation="chat")] == ["new", "old"]
    assert [t.trace_id for t in store.recent(1)] == ["judged"]


def test_feedback_attaches_to_known_traces_only(tmp_path: Path) -> None:
    store = SqliteTraceStore(tmp_path / "t.sqlite3")
    store.record(trace("a"))

    assert store.add_feedback(Feedback("a", Rating.DOWN, "wrong fee")) is True
    assert store.add_feedback(Feedback("nope", Rating.UP)) is False
    assert store.add_feedback(Feedback("a", Rating.UP)) is True  # a user can change their mind

    found = store.feedback_for(["a", "nope"])
    assert list(found) == ["a"]
    assert found["a"].rating is Rating.UP
    assert store.feedback_for([]) == {}


def test_summary_reports_latency_errors_cost_and_feedback() -> None:
    traces = [trace(str(i), ms=ms) for i, ms in enumerate([100, 200, 300, 400])]
    traces.append(trace("x", status=TraceStatus.ERROR, ms=9999))
    summary = summarize(traces, {"0": Feedback("0", Rating.UP), "1": Feedback("1", Rating.DOWN)})

    assert summary.count == 5
    assert summary.errors == 1
    assert summary.error_rate == 0.2
    assert (summary.p50_ms, summary.p95_ms) == (200, 400)  # failed requests are excluded
    assert summary.cost_usd == 0.005
    assert (summary.thumbs_up, summary.thumbs_down) == (1, 1)

    assert summarize([trace("a"), trace("b", cost=None)]).cost_usd is None


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([], 0.5) == 0
    assert percentile([7], 0.95) == 7
    assert percentile(list(range(1, 101)), 0.5) == 50
    assert percentile(list(range(1, 101)), 0.95) == 95
