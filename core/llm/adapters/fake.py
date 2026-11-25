"""Deterministic provider for tests and for running the service with no model at all."""

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from core.config import ModelSpec
from core.llm.errors import LLMError
from core.llm.types import (
    Completion,
    LLMRequest,
    StopReason,
    StreamEvent,
    TextDelta,
    ToolCall,
    Usage,
)

# A scripted reply: text, or tool calls the model "asks for".
Reply = str | tuple[ToolCall, ...]


class FakeProvider:
    """Answers without a model.

    By default it echoes the last user message, or, when the request asks for structured
    output, returns a minimal object that fits the schema. Tests can instead script fixed
    replies (used in order, the last one repeating), including tool calls, or a fixed error.
    """

    def __init__(
        self,
        spec: ModelSpec,
        reply: str | None = None,
        replies: Sequence[Reply] = (),
        error: LLMError | None = None,
    ) -> None:
        self._spec = spec
        self._replies: list[Reply] = (
            list(replies) if replies else ([reply] if reply is not None else [])
        )
        self._error = error
        self.requests: list[LLMRequest] = []  # lets tests assert on what was sent

    @property
    def name(self) -> str:
        return f"{self._spec.provider.value}:{self._spec.model}"

    async def complete(self, request: LLMRequest) -> Completion:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        reply = self._next_reply(request)
        input_tokens = sum(len(m.content.split()) for m in request.messages)
        if isinstance(reply, tuple):
            usage = Usage(input_tokens, output_tokens=len(reply))
            return Completion("", self._spec, usage, StopReason.TOOL_USE, 0, tool_calls=reply)
        usage = Usage(input_tokens, output_tokens=len(reply.split()))
        return Completion(reply, self._spec, usage, StopReason.END, latency_ms=0)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        completion = await self.complete(request)
        if completion.text:
            for word in completion.text.split(" "):
                yield TextDelta(word + " ")
        yield completion

    def _next_reply(self, request: LLMRequest) -> Reply:
        if self._replies:
            index = min(len(self.requests) - 1, len(self._replies) - 1)
            return self._replies[index]
        if request.json_schema is not None:
            return json.dumps(_sample(request.json_schema, request.json_schema))
        return f"echo: {request.messages[-1].content}"


def _sample(schema: dict[str, Any], root: dict[str, Any]) -> Any:
    """Smallest value that fits ``schema`` (enough for the schemas the core produces)."""
    if "$ref" in schema:
        return _sample(root["$defs"][schema["$ref"].rsplit("/", 1)[-1]], root)
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        return _sample(schema["anyOf"][0], root)
    kind = schema.get("type")
    if kind == "object":
        return {key: _sample(sub, root) for key, sub in schema.get("properties", {}).items()}
    if kind == "array":
        return []
    return {"string": "fake", "integer": 0, "number": 0, "boolean": False}.get(str(kind))
