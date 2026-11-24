"""Small builders shared by the model layer tests."""

from collections.abc import AsyncIterator

from core.config import ModelSpec, Provider
from core.llm import (
    Completion,
    LLMError,
    LLMRequest,
    Message,
    Role,
    StopReason,
    StreamEvent,
    TextDelta,
    Usage,
)


def spec(provider: Provider = Provider.FAKE, model: str = "m1") -> ModelSpec:
    return ModelSpec(provider=provider, model=model)


def request(text: str = "hello", system: str = "be brief") -> LLMRequest:
    return LLMRequest(system=system, messages=(Message(Role.USER, text),))


async def collect(stream: AsyncIterator[StreamEvent]) -> tuple[str, Completion]:
    """Drain a stream and return (joined delta text, final completion)."""
    parts: list[str] = []
    final: Completion | None = None
    async for event in stream:
        if isinstance(event, TextDelta):
            assert final is None, "a delta arrived after the final completion"
            parts.append(event.text)
        else:
            final = event
    assert final is not None, "the stream ended without a completion"
    return "".join(parts), final


class BrokenStreamProvider:
    """Streams some deltas and then fails, or fails before the first delta."""

    def __init__(self, name: str, deltas_before_error: int, error: LLMError) -> None:
        self._name = name
        self._deltas = deltas_before_error
        self._error = error

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, request: LLMRequest) -> Completion:
        raise self._error

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        for index in range(self._deltas):
            yield TextDelta(f"part{index} ")
        raise self._error


def completion(text: str = "ok") -> Completion:
    return Completion(text, spec(), Usage(1, 1), StopReason.END, 0)
