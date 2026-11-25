"""Trace value objects. A trace is the full record of one request to the brain."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

AttributeValue = str | int | float | bool | None
Attributes = Mapping[str, AttributeValue]


class TraceStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class Rating(StrEnum):
    UP = "up"
    DOWN = "down"


@dataclass(frozen=True, slots=True)
class Span:
    """One timed step inside a request, such as retrieval or the model call."""

    name: str
    offset_ms: int  # start, relative to the start of the trace
    duration_ms: int
    attributes: Attributes = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Event:
    """Something that happened at a point in time, such as a provider failing over."""

    name: str
    offset_ms: int
    attributes: Attributes = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Trace:
    """Everything needed to explain, cost and replay one answer."""

    trace_id: str
    operation: str  # what was asked of the brain: chat, title, eval, ...
    started_at: datetime
    duration_ms: int
    status: TraceStatus
    error: str | None = None
    provider: str | None = None  # who actually answered, after any failover
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # None when the model has no price, which is not zero
    prompts: tuple[str, ...] = ()  # name@vN#checksum
    retrieved: tuple[str, ...] = ()  # chunk ids given to the model, in rank order
    cited: tuple[str, ...] = ()  # the subset the answer cited
    spans: tuple[Span, ...] = ()
    events: tuple[Event, ...] = ()
    attributes: Attributes = field(default_factory=dict)
    input: str | None = None  # kept only when the pack allows capturing content
    output: str | None = None


@dataclass(frozen=True, slots=True)
class Feedback:
    """A user's verdict on one answer, joined to its trace by ``trace_id``."""

    trace_id: str
    rating: Rating
    comment: str = ""
    created_at: datetime | None = None
