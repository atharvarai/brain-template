"""Aggregates over a set of traces: what an on-call engineer looks at first."""

from collections.abc import Sequence
from dataclasses import dataclass

from core.observability.types import Feedback, Rating, Trace, TraceStatus


def percentile(values: Sequence[int], fraction: float) -> int:
    """Nearest-rank percentile. 0 for an empty list."""
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, round(fraction * len(ordered) + 0.5 - 1e-9))
    return ordered[min(rank, len(ordered)) - 1]


@dataclass(frozen=True, slots=True)
class TraceSummary:
    count: int
    errors: int
    p50_ms: int
    p95_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None if any traced call had no known price
    thumbs_up: int
    thumbs_down: int

    @property
    def error_rate(self) -> float:
        return self.errors / self.count if self.count else 0.0


def summarize(traces: Sequence[Trace], feedback: dict[str, Feedback] | None = None) -> TraceSummary:
    feedback = feedback or {}
    latencies = [t.duration_ms for t in traces if t.status is TraceStatus.OK]
    costs = [t.cost_usd for t in traces if t.model is not None]
    ratings = [f.rating for f in feedback.values()]
    return TraceSummary(
        count=len(traces),
        errors=sum(t.status is TraceStatus.ERROR for t in traces),
        p50_ms=percentile(latencies, 0.5),
        p95_ms=percentile(latencies, 0.95),
        input_tokens=sum(t.input_tokens for t in traces),
        output_tokens=sum(t.output_tokens for t in traces),
        cost_usd=None if any(c is None for c in costs) else sum(c for c in costs if c),
        thumbs_up=ratings.count(Rating.UP),
        thumbs_down=ratings.count(Rating.DOWN),
    )
