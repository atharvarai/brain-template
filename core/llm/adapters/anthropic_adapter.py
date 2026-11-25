"""Adapter for the Anthropic Messages API. Reads ANTHROPIC_API_KEY from the environment."""

import time
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic.types import ContentBlockParam, MessageParam, ToolParam
from anthropic.types import Message as AnthropicMessage

from core.config import ModelSpec
from core.llm.errors import ProviderUnavailableError, error_for_status
from core.llm.types import (
    Completion,
    LLMRequest,
    Role,
    StopReason,
    StreamEvent,
    TextDelta,
    ToolCall,
    Usage,
)

_STOP_REASONS = {
    "end_turn": StopReason.END,
    "stop_sequence": StopReason.END,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
    "tool_use": StopReason.TOOL_USE,
}


class AnthropicProvider:
    def __init__(
        self,
        spec: ModelSpec,
        client: anthropic.AsyncAnthropic | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._spec = spec
        self._client = client or anthropic.AsyncAnthropic(max_retries=2, timeout=timeout_seconds)

    @property
    def name(self) -> str:
        return f"{self._spec.provider.value}:{self._spec.model}"

    async def complete(self, request: LLMRequest) -> Completion:
        start = time.perf_counter()
        try:
            message = await self._client.messages.create(
                model=self._spec.model,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=_to_params(request),
                **_structured_output(request),
                **_tools(request),
            )
        except anthropic.APIConnectionError as exc:  # includes timeouts
            raise ProviderUnavailableError(f"{self.name}: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise error_for_status(self.name, exc.status_code, exc.message) from exc
        return self._completion(message, start)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        start = time.perf_counter()
        try:
            async with self._client.messages.stream(
                model=self._spec.model,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=_to_params(request),
                **_structured_output(request),
                **_tools(request),
            ) as stream:
                async for text in stream.text_stream:
                    yield TextDelta(text)
                message = await stream.get_final_message()
        except anthropic.APIConnectionError as exc:
            raise ProviderUnavailableError(f"{self.name}: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise error_for_status(self.name, exc.status_code, exc.message) from exc
        yield self._completion(message, start)

    def _completion(self, message: AnthropicMessage, start: float) -> Completion:
        usage = message.usage
        input_tokens = (
            usage.input_tokens
            + (usage.cache_creation_input_tokens or 0)
            + (usage.cache_read_input_tokens or 0)
        )
        return Completion(
            text="".join(block.text for block in message.content if block.type == "text"),
            spec=self._spec,
            usage=Usage(input_tokens, usage.output_tokens),
            stop_reason=_STOP_REASONS.get(str(message.stop_reason), StopReason.OTHER),
            latency_ms=int((time.perf_counter() - start) * 1000),
            tool_calls=tuple(
                ToolCall(block.id, block.name, dict(block.input))
                for block in message.content
                if block.type == "tool_use"
            ),
        )


def _structured_output(request: LLMRequest) -> dict[str, Any]:
    if request.json_schema is None:
        return {}
    return {"output_config": {"format": {"type": "json_schema", "schema": request.json_schema}}}


def _tools(request: LLMRequest) -> dict[str, Any]:
    # Tool definitions are sent even when calls are not allowed: the API requires them
    # whenever the history holds tool calls.
    if not request.tools:
        return {}
    tools: list[ToolParam] = [
        {"name": t.name, "description": t.description, "input_schema": dict(t.input_schema)}
        for t in request.tools
    ]
    return {"tools": tools, "tool_choice": {"type": request.tool_choice.value}}


def _to_params(request: LLMRequest) -> list[MessageParam]:
    params: list[MessageParam] = []
    for message in request.messages:
        if not message.tool_calls and not message.tool_results:
            if message.role is Role.USER:
                params.append({"role": "user", "content": message.content})
            else:
                params.append({"role": "assistant", "content": message.content})
            continue
        blocks: list[ContentBlockParam] = [
            {
                "type": "tool_result",
                "tool_use_id": result.call_id,
                "content": result.content,
                "is_error": result.is_error,
            }
            for result in message.tool_results
        ]
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        blocks.extend(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": dict(call.arguments)}
            for call in message.tool_calls
        )
        if message.role is Role.USER:
            params.append({"role": "user", "content": blocks})
        else:
            params.append({"role": "assistant", "content": blocks})
    return params
