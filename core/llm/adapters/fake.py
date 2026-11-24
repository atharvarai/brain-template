"""Deterministic provider for tests and for running the service with no model at all."""

from collections.abc import AsyncIterator

from core.config import ModelSpec
from core.llm.errors import LLMError
from core.llm.types import Completion, LLMRequest, StopReason, StreamEvent, TextDelta, Usage


class FakeProvider:
    """Echoes the last user message, or returns a fixed reply, or raises a fixed error."""

    def __init__(
        self, spec: ModelSpec, reply: str | None = None, error: LLMError | None = None
    ) -> None:
        self._spec = spec
        self._reply = reply
        self._error = error
        self.requests: list[LLMRequest] = []  # lets tests assert on what was sent

    @property
    def name(self) -> str:
        return f"{self._spec.provider.value}:{self._spec.model}"

    async def complete(self, request: LLMRequest) -> Completion:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return self._completion(request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        completion = await self.complete(request)
        for word in completion.text.split(" "):
            yield TextDelta(word + " ")
        yield completion

    def _completion(self, request: LLMRequest) -> Completion:
        text = self._reply if self._reply is not None else f"echo: {request.messages[-1].content}"
        usage = Usage(
            input_tokens=sum(len(m.content.split()) for m in request.messages),
            output_tokens=len(text.split()),
        )
        return Completion(text, self._spec, usage, StopReason.END, latency_ms=0)
