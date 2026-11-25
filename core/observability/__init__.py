"""Layer 08, observability: one trace per request (timings, tokens, cost, prompt versions,
documents used), user feedback joined to traces, and summaries over them."""

from core.observability.pricing import PriceBook
from core.observability.serialize import trace_from_dict, trace_to_dict
from core.observability.sinks import (
    FanOutSink,
    FeedbackSink,
    InMemoryTraceStore,
    LoggingTraceSink,
    TraceReader,
    TraceSink,
)
from core.observability.sqlite_store import SqliteTraceStore
from core.observability.summary import TraceSummary, percentile, summarize
from core.observability.tracer import ActiveTrace, Tracer, current_trace, record_event
from core.observability.types import (
    Attributes,
    AttributeValue,
    Event,
    Feedback,
    Rating,
    Span,
    Trace,
    TraceStatus,
)

__all__ = [
    "ActiveTrace",
    "AttributeValue",
    "Attributes",
    "Event",
    "FanOutSink",
    "Feedback",
    "FeedbackSink",
    "InMemoryTraceStore",
    "LoggingTraceSink",
    "PriceBook",
    "Rating",
    "Span",
    "SqliteTraceStore",
    "Trace",
    "TraceReader",
    "TraceSink",
    "TraceStatus",
    "TraceSummary",
    "Tracer",
    "current_trace",
    "percentile",
    "record_event",
    "summarize",
    "trace_from_dict",
    "trace_to_dict",
]
