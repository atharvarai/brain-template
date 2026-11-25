"""Where traces go. Split into narrow ports so a write-only backend needs no query code."""

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from core.observability.types import Feedback, Trace

logger = logging.getLogger(__name__)


class TraceSink(Protocol):
    def record(self, trace: Trace) -> None: ...


class TraceReader(Protocol):
    def recent(self, limit: int, operation: str | None = None) -> list[Trace]:
        """Newest first."""
        ...

    def get(self, trace_id: str) -> Trace | None: ...

    def feedback_for(self, trace_ids: Sequence[str]) -> dict[str, Feedback]: ...


class FeedbackSink(Protocol):
    def add_feedback(self, feedback: Feedback) -> bool:
        """Store feedback. Returns False if no trace has that id."""
        ...


class LoggingTraceSink:
    """One structured log line per trace, for log-based pipelines (Loki, CloudWatch...)."""

    def record(self, trace: Trace) -> None:
        summary = {
            "trace_id": trace.trace_id,
            "operation": trace.operation,
            "status": trace.status.value,
            "provider": trace.provider,
            "model": trace.model,
            "ms": trace.duration_ms,
            "tokens_in": trace.input_tokens,
            "tokens_cached": trace.cache_read_tokens,
            "tokens_out": trace.output_tokens,
            "cost_usd": trace.cost_usd,
            "prompts": trace.prompts,
            "retrieved": trace.retrieved,
            "cited": trace.cited,
        }
        if trace.error:
            summary["error"] = trace.error
        logger.info("trace %s", json.dumps(summary))


class FanOutSink:
    """Composite: sends each trace to several sinks. One failing sink does not stop the rest."""

    def __init__(self, sinks: Sequence[TraceSink]) -> None:
        self._sinks = tuple(sinks)

    def record(self, trace: Trace) -> None:
        for sink in self._sinks:
            try:
                sink.record(trace)
            except Exception:
                logger.exception("trace sink %s failed", type(sink).__name__)


class InMemoryTraceStore:
    """Keeps traces in a list. For tests and for embedding the brain in a script."""

    def __init__(self) -> None:
        self.traces: list[Trace] = []
        self.feedback: dict[str, Feedback] = {}

    def record(self, trace: Trace) -> None:
        self.traces.append(trace)

    def recent(self, limit: int, operation: str | None = None) -> list[Trace]:
        matching = [t for t in reversed(self.traces) if operation in (None, t.operation)]
        return matching[:limit]

    def get(self, trace_id: str) -> Trace | None:
        return next((t for t in self.traces if t.trace_id == trace_id), None)

    def feedback_for(self, trace_ids: Sequence[str]) -> dict[str, Feedback]:
        return {tid: self.feedback[tid] for tid in trace_ids if tid in self.feedback}

    def add_feedback(self, feedback: Feedback) -> bool:
        if self.get(feedback.trace_id) is None:
            return False
        created = feedback.created_at or datetime.now(UTC)
        self.feedback[feedback.trace_id] = Feedback(
            feedback.trace_id, feedback.rating, feedback.comment, created
        )
        return True
