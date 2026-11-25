"""Adapter for the OpenAI Chat Completions API. Reads OPENAI_API_KEY from the environment."""

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import openai
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
)

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
    "stop": StopReason.END,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
    "tool_calls": StopReason.TOOL_USE,
}


class OpenAIProvider:
    def __init__(
        self,
        spec: ModelSpec,
        client: openai.AsyncOpenAI | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._spec = spec
        self._client = client or openai.AsyncOpenAI(max_retries=2, timeout=timeout_seconds)

    @property
    def name(self) -> str:
        return f"{self._spec.provider.value}:{self._spec.model}"

    async def complete(self, request: LLMRequest) -> Completion:
        start = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self._spec.model,
                messages=_to_params(request),
                max_completion_tokens=request.max_tokens,
                **_structured_output(request),
                **_tools(request),
            )
        except openai.APIConnectionError as exc:  # includes timeouts
            raise ProviderUnavailableError(f"{self.name}: {exc}") from exc
        except openai.APIStatusError as exc:
            raise error_for_status(self.name, exc.status_code, exc.message) from exc

        choice = response.choices[0]
        usage = response.usage
        calls = tuple(
            _tool_call(call.id, call.function.name, call.function.arguments)
            for call in choice.message.tool_calls or ()
            if call.type == "function"
        )
        return self._completion(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=Usage(usage.prompt_tokens, usage.completion_tokens) if usage else Usage(),
            start=start,
            tool_calls=calls,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        start = time.perf_counter()
        parts: list[str] = []
        finish_reason: str | None = None
        usage = Usage()
        calls: dict[int, _PartialCall] = {}  # tool calls arrive in pieces, keyed by index
        try:
            chunks = await self._client.chat.completions.create(
                model=self._spec.model,
                messages=_to_params(request),
                max_completion_tokens=request.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                **_structured_output(request),
                **_tools(request),
            )
            async for chunk in chunks:
                if chunk.usage:  # arrives on the last chunk, which has no choices
                    usage = Usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                for choice in chunk.choices:
                    if choice.delta.content:
                        parts.append(choice.delta.content)
                        yield TextDelta(choice.delta.content)
                    for piece in choice.delta.tool_calls or ():
                        partial = calls.setdefault(piece.index, _PartialCall())
                        if piece.id:
                            partial.id = piece.id
                        if piece.function and piece.function.name:
                            partial.name += piece.function.name
                        if piece.function and piece.function.arguments:
                            partial.arguments += piece.function.arguments
                    finish_reason = choice.finish_reason or finish_reason
        except openai.APIConnectionError as exc:
            raise ProviderUnavailableError(f"{self.name}: {exc}") from exc
        except openai.APIStatusError as exc:
            raise error_for_status(self.name, exc.status_code, exc.message) from exc
        tool_calls = tuple(_tool_call(c.id, c.name, c.arguments) for _, c in sorted(calls.items()))
        yield self._completion("".join(parts), finish_reason, usage, start, tool_calls)

    def _completion(
        self,
        text: str,
        finish_reason: str | None,
        usage: Usage,
        start: float,
        tool_calls: tuple[ToolCall, ...] = (),
    ) -> Completion:
        return Completion(
            text=text,
            spec=self._spec,
            usage=usage,
            stop_reason=_STOP_REASONS.get(str(finish_reason), StopReason.OTHER),
            latency_ms=int((time.perf_counter() - start) * 1000),
            tool_calls=tool_calls,
        )


@dataclass
class _PartialCall:
    id: str = ""
    name: str = ""
    arguments: str = ""


def _tool_call(call_id: str, name: str, arguments: str) -> ToolCall:
    """Arguments arrive as a JSON string. If the model wrote invalid JSON, the call goes
    ahead with no arguments and the tool's own validation error tells the model why."""
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        parsed = {}
    return ToolCall(call_id, name, parsed if isinstance(parsed, dict) else {})


def _tools(request: LLMRequest) -> dict[str, Any]:
    if not request.tools:
        return {}
    tools: list[ChatCompletionFunctionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": dict(t.input_schema),
            },
        }
        for t in request.tools
    ]
    return {"tools": tools, "tool_choice": request.tool_choice.value}


def _structured_output(request: LLMRequest) -> dict[str, Any]:
    if request.json_schema is None:
        return {}
    json_schema = {"name": "output", "schema": request.json_schema, "strict": True}
    return {"response_format": {"type": "json_schema", "json_schema": json_schema}}


def _to_params(request: LLMRequest) -> list[ChatCompletionMessageParam]:
    params: list[ChatCompletionMessageParam] = [{"role": "system", "content": request.system}]
    for message in request.messages:
        if message.role is Role.USER:
            # Tool results are separate "tool" messages, answering the calls in order.
            for result in message.tool_results:
                content = f"Error: {result.content}" if result.is_error else result.content
                params.append({"role": "tool", "tool_call_id": result.call_id, "content": content})
            if message.content or not message.tool_results:
                params.append({"role": "user", "content": message.content})
        elif message.tool_calls:
            calls: list[ChatCompletionMessageFunctionToolCallParam] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(dict(call.arguments))},
                }
                for call in message.tool_calls
            ]
            params.append(
                {"role": "assistant", "content": message.content or None, "tool_calls": calls}
            )
        else:
            params.append({"role": "assistant", "content": message.content})
    return params
