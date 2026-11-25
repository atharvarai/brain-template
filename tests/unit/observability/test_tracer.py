"""Tracer: spans, cost, errors, content capture and events from deep layers."""

from datetime import UTC, datetime

import pytest

from core.config import ModelSpec, Price, Provider
from core.llm import Completion, StopReason, Usage
from core.observability import (
    FanOutSink,
    InMemoryTraceStore,
    PriceBook,
    Trace,
    Tracer,
    TraceStatus,
    current_trace,
    record_event,
)

pytestmark = pytest.mark.anyio


class Clock:
    """Advances only when told to, so durations are exact."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, ms: int) -> None:
        self.now += ms / 1000


def completion(model: str = "m1", tokens_in: int = 1000, tokens_out: int = 200) -> Completion:
    spec = ModelSpec(provider=Provider.FAKE, model=model)
    return Completion("hi", spec, Usage(tokens_in, tokens_out), StopReason.END, 5)


def tracer(store: InMemoryTraceStore, clock: Clock | None = None, capture: bool = False) -> Tracer:
    prices = PriceBook({"m1": Price(input=2.0, output=10.0)})
    return Tracer(
        store,
        prices,
        capture_content=capture,
        attributes={"pack": "Acme"},
        clock=clock or Clock(),
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        new_id=lambda: "t1",
    )


async def test_a_trace_records_spans_tokens_cost_and_attributes() -> None:
    store, clock = InMemoryTraceStore(), Clock()
    async with tracer(store, clock).trace("chat") as trace:
        with trace.span("retrieve", k=5) as span:
            clock.advance(30)
            span["hits"] = 3
        with trace.span("generate"):
            clock.advance(400)
        trace.record_completion(completion())
        trace.record_prompts(["system@v2#abc"])
        trace.record_retrieval(["a.md#0", "b.md#1"], ["a.md#0"])

    [saved] = store.traces
    assert saved.trace_id == "t1"
    assert saved.status is TraceStatus.OK
    assert saved.duration_ms == 430
    assert [(s.name, s.offset_ms, s.duration_ms) for s in saved.spans] == [
        ("retrieve", 0, 30),
        ("generate", 30, 400),
    ]
    assert saved.spans[0].attributes == {"k": 5, "hits": 3}
    assert (saved.provider, saved.model) == ("fake", "m1")
    assert (saved.input_tokens, saved.output_tokens) == (1000, 200)
    assert saved.cost_usd == pytest.approx((1000 * 2 + 200 * 10) / 1_000_000)
    assert saved.prompts == ("system@v2#abc",)
    assert saved.cited == ("a.md#0",)
    assert saved.attributes == {"pack": "Acme"}


async def test_several_calls_add_up_and_an_unpriced_model_makes_cost_unknown() -> None:
    store = InMemoryTraceStore()
    async with tracer(store).trace("eval") as trace:
        trace.record_completion(completion())
        trace.record_completion(completion())
    assert store.traces[0].input_tokens == 2000
    assert store.traces[0].cost_usd == pytest.approx(0.008)

    async with tracer(store).trace("eval") as trace:
        trace.record_completion(completion())
        trace.record_completion(completion(model="no-price"))
    assert store.traces[1].cost_usd is None


async def test_a_trace_with_no_model_call_has_no_cost() -> None:
    store = InMemoryTraceStore()
    async with tracer(store).trace("chat"):
        pass
    assert store.traces[0].cost_usd is None


async def test_a_failed_request_is_still_traced_and_the_error_propagates() -> None:
    store = InMemoryTraceStore()
    with pytest.raises(RuntimeError):
        async with tracer(store).trace("chat") as trace:
            with trace.span("generate"):
                raise RuntimeError("boom")
    [saved] = store.traces
    assert saved.status is TraceStatus.ERROR
    assert saved.error == "RuntimeError: boom"
    assert [s.name for s in saved.spans] == ["generate"]  # the failing step is kept


async def test_content_is_kept_only_when_the_pack_allows_it() -> None:
    store = InMemoryTraceStore()
    async with tracer(store).trace("chat") as trace:
        trace.capture(input="my phone is 98765", output="ok")
    async with tracer(store, capture=True).trace("chat") as trace:
        trace.capture(input="question", output="answer")
    assert (store.traces[0].input, store.traces[0].output) == (None, None)
    assert (store.traces[1].input, store.traces[1].output) == ("question", "answer")


async def test_deep_layers_add_events_to_the_active_trace_only() -> None:
    store = InMemoryTraceStore()
    record_event("ignored")  # no active trace: a no-op, not an error
    async with tracer(store).trace("chat"):
        assert current_trace() is not None
        record_event("provider_failed", provider="a")
    assert current_trace() is None
    assert [(e.name, dict(e.attributes)) for e in store.traces[0].events] == [
        ("provider_failed", {"provider": "a"})
    ]


class BrokenSink:
    def record(self, trace: Trace) -> None:
        raise OSError("disk full")


async def test_a_broken_sink_never_breaks_the_request() -> None:
    async with Tracer(BrokenSink()).trace("chat"):
        pass  # no exception escapes

    good = InMemoryTraceStore()
    async with Tracer(FanOutSink([BrokenSink(), good])).trace("chat"):
        pass
    assert len(good.traces) == 1  # the composite still feeds the healthy sink
