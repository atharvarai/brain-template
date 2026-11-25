"""Builds one trace per request and hands it to a sink when the request ends.

Usage::

    async with tracer.trace("chat") as trace:
        with trace.span("retrieve"):
            ...
        trace.record_completion(completion)

Layers deep in the stack (for example provider failover) call ``record_event`` and the
event lands on whatever trace is active, without being handed the trace object.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Protocol

from core.observability.pricing import PriceBook
from core.observability.sinks import TraceSink
from core.observability.types import (
    Attributes,
    AttributeValue,
    Event,
    Span,
    Trace,
    TraceStatus,
)

logger = logging.getLogger(__name__)
_ERROR_CHARS = 500


class CompletionLike(Protocol):
    """What a trace needs from a model answer. Structural, so observability stays below
    the model layer instead of importing it."""

    @property
    def spec(self) -> "SpecLike": ...

    @property
    def usage(self) -> "UsageLike": ...


class SpecLike(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...


class UsageLike(Protocol):
    @property
    def input_tokens(self) -> int: ...

    @property
    def output_tokens(self) -> int: ...


class ActiveTrace:
    """A trace being recorded. Mutable on purpose; it becomes an immutable ``Trace`` at the end."""

    def __init__(
        self,
        trace_id: str,
        operation: str,
        prices: PriceBook,
        capture_content: bool,
        clock: Callable[[], float],
        started_at: datetime,
        attributes: Attributes,
    ) -> None:
        self.trace_id = trace_id
        self._operation = operation
        self._prices = prices
        self._capture = capture_content
        self._clock = clock
        self._start = clock()
        self._started_at = started_at
        self._attributes: dict[str, AttributeValue] = dict(attributes)
        self._spans: list[Span] = []
        self._events: list[Event] = []
        self._provider: str | None = None
        self._model: str | None = None
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost: float | None = 0.0
        self._prompts: tuple[str, ...] = ()
        self._retrieved: tuple[str, ...] = ()
        self._cited: tuple[str, ...] = ()
        self._input: str | None = None
        self._output: str | None = None

    def _elapsed_ms(self) -> int:
        return round((self._clock() - self._start) * 1000)

    @contextmanager
    def span(self, name: str, **attributes: AttributeValue) -> Iterator[dict[str, AttributeValue]]:
        """Time a step. The yielded dict can take attributes known only at the end."""
        offset = self._elapsed_ms()
        values: dict[str, AttributeValue] = dict(attributes)
        try:
            yield values
        finally:
            duration = self._elapsed_ms() - offset
            self._spans.append(Span(name, offset, duration, values))

    def event(self, name: str, **attributes: AttributeValue) -> None:
        self._events.append(Event(name, self._elapsed_ms(), attributes))

    def set(self, **attributes: AttributeValue) -> None:
        self._attributes.update(attributes)

    def record_completion(self, completion: CompletionLike) -> None:
        """Add one model call. Tokens and cost add up when a request makes several calls."""
        spec, usage = completion.spec, completion.usage
        self._provider, self._model = str(spec.provider), spec.model
        self._input_tokens += usage.input_tokens
        self._output_tokens += usage.output_tokens
        cost = self._prices.cost(spec.model, usage.input_tokens, usage.output_tokens)
        self._cost = None if cost is None or self._cost is None else self._cost + cost

    def record_prompts(self, refs: Iterable[str]) -> None:
        self._prompts = tuple(refs)

    def record_retrieval(self, retrieved: Iterable[str], cited: Iterable[str] = ()) -> None:
        self._retrieved, self._cited = tuple(retrieved), tuple(cited)

    def capture(self, *, input: str | None = None, output: str | None = None) -> None:
        """Keep the request and answer text, if the pack allows it. Otherwise a no-op."""
        if not self._capture:
            return
        if input is not None:
            self._input = input
        if output is not None:
            self._output = output

    def finish(self, status: TraceStatus, error: str | None = None) -> Trace:
        called = self._model is not None
        return Trace(
            trace_id=self.trace_id,
            operation=self._operation,
            started_at=self._started_at,
            duration_ms=self._elapsed_ms(),
            status=status,
            error=error,
            provider=self._provider,
            model=self._model,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cost_usd=self._cost if called else None,
            prompts=self._prompts,
            retrieved=self._retrieved,
            cited=self._cited,
            spans=tuple(self._spans),
            events=tuple(self._events),
            attributes=dict(self._attributes),
            input=self._input,
            output=self._output,
        )


_active: ContextVar[ActiveTrace | None] = ContextVar("active_trace", default=None)


def current_trace() -> ActiveTrace | None:
    return _active.get()


def record_event(name: str, **attributes: AttributeValue) -> None:
    """Attach an event to the active trace, if any. Safe to call from any layer."""
    active = _active.get()
    if active is not None:
        active.event(name, **attributes)


def _describe(exc: BaseException) -> str:
    if isinstance(exc, GeneratorExit | asyncio.CancelledError):
        return "cancelled"
    return f"{type(exc).__name__}: {exc}"[:_ERROR_CHARS]


class Tracer:
    def __init__(
        self,
        sink: TraceSink,
        prices: PriceBook | None = None,
        capture_content: bool = False,
        attributes: Mapping[str, AttributeValue] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        new_id: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._sink = sink
        self._prices = prices or PriceBook({})
        self._capture = capture_content
        self._attributes = dict(attributes or {})
        self._clock = clock
        self._now = now
        self._new_id = new_id

    @asynccontextmanager
    async def trace(self, operation: str) -> AsyncIterator[ActiveTrace]:
        active = ActiveTrace(
            self._new_id(),
            operation,
            self._prices,
            self._capture,
            self._clock,
            self._now(),
            self._attributes,
        )
        token = _active.set(active)
        status, error = TraceStatus.OK, None
        try:
            yield active
        except BaseException as exc:
            status, error = TraceStatus.ERROR, _describe(exc)
            raise
        finally:
            try:
                _active.reset(token)
            except ValueError:  # closed from another context (e.g. an abandoned stream)
                _active.set(None)
            self._emit(active.finish(status, error))

    def _emit(self, trace: Trace) -> None:
        # Observability must never take the product down: a failing sink is only logged.
        try:
            self._sink.record(trace)
        except Exception:
            logger.exception("could not record trace %s", trace.trace_id)
