"""Adapter for a local Ollama server (free, offline). Base URL comes from OLLAMA_HOST."""

import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx2

from core.config import ModelSpec
from core.llm.errors import ProviderUnavailableError, error_for_status
from core.llm.types import (
    Completion,
    LLMRequest,
    Message,
    Role,
    StopReason,
    StreamEvent,
    TextDelta,
    ToolCall,
    Usage,
)

DEFAULT_BASE_URL = "http://localhost:11434"
_STOP_REASONS = {"stop": StopReason.END, "length": StopReason.MAX_TOKENS}


class OllamaProvider:
    def __init__(
        self,
        spec: ModelSpec,
        client: httpx2.AsyncClient | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._spec = spec
        self._client = client or httpx2.AsyncClient(timeout=timeout_seconds)
        self._url = (base_url or os.environ.get("OLLAMA_HOST") or DEFAULT_BASE_URL).rstrip("/")

    @property
    def name(self) -> str:
        return f"{self._spec.provider.value}:{self._spec.model}"

    async def complete(self, request: LLMRequest) -> Completion:
        start = time.perf_counter()
        try:
            response = await self._client.post(
                f"{self._url}/api/chat", json=self._payload(request, stream=False)
            )
            response.raise_for_status()
        except httpx2.HTTPStatusError as exc:
            raise error_for_status(
                self.name, exc.response.status_code, exc.response.text[:200]
            ) from exc
        except httpx2.HTTPError as exc:  # connection refused, timeout
            raise ProviderUnavailableError(f"{self.name}: {exc!r}") from exc
        data = response.json()
        message = data["message"]
        return self._completion(message["content"], data, start, _tool_calls(message))

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        start = time.perf_counter()
        parts: list[str] = []
        calls: list[ToolCall] = []
        final: dict[str, Any] = {}
        try:
            async with self._client.stream(
                "POST", f"{self._url}/api/chat", json=self._payload(request, stream=True)
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")
                    raise error_for_status(self.name, response.status_code, body[:200])
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    message = data.get("message", {})
                    if text := message.get("content"):
                        parts.append(text)
                        yield TextDelta(text)
                    calls.extend(_tool_calls(message, offset=len(calls)))
                    if data.get("done"):
                        final = data
        except httpx2.HTTPError as exc:
            raise ProviderUnavailableError(f"{self.name}: {exc!r}") from exc
        yield self._completion("".join(parts), final, start, tuple(calls))

    def _payload(self, request: LLMRequest, stream: bool) -> dict[str, Any]:
        structured = {"format": request.json_schema} if request.json_schema else {}
        # Ollama has no tool_choice: tools are simply left out when calls are not allowed.
        tools = (
            {
                "tools": [
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
            }
            if request.offers_tools
            else {}
        )
        messages = [{"role": "system", "content": request.system}]
        for message in request.messages:
            messages.extend(_to_messages(message))
        return {
            **structured,
            **tools,
            "model": self._spec.model,
            "messages": messages,
            "stream": stream,
            "options": {"num_predict": request.max_tokens},
        }

    def _completion(
        self,
        text: str,
        data: dict[str, Any],
        start: float,
        tool_calls: tuple[ToolCall, ...] = (),
    ) -> Completion:
        # Ollama reports "stop" even when the model asked for tools.
        stop = _STOP_REASONS.get(str(data.get("done_reason")), StopReason.OTHER)
        return Completion(
            text=text,
            spec=self._spec,
            usage=Usage(int(data.get("prompt_eval_count", 0)), int(data.get("eval_count", 0))),
            stop_reason=StopReason.TOOL_USE if tool_calls else stop,
            latency_ms=int((time.perf_counter() - start) * 1000),
            tool_calls=tool_calls,
        )


def _tool_calls(message: dict[str, Any], offset: int = 0) -> tuple[ToolCall, ...]:
    """Ollama gives calls no ids, so they are numbered in the order they arrive."""
    calls = []
    for index, call in enumerate(message.get("tool_calls") or (), start=offset):
        function = call.get("function") or {}
        arguments = function.get("arguments")
        calls.append(
            ToolCall(
                id=str(call.get("id") or f"call_{index}"),
                name=str(function.get("name", "")),
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
    return tuple(calls)


def _to_messages(message: Message) -> list[dict[str, Any]]:
    if message.role is Role.USER:
        results: list[dict[str, Any]] = [
            {
                "role": "tool",
                "tool_name": r.name,
                "content": f"Error: {r.content}" if r.is_error else r.content,
            }
            for r in message.tool_results
        ]
        if message.content or not message.tool_results:
            results.append({"role": "user", "content": message.content})
        return results
    reply: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        reply["tool_calls"] = [
            {"function": {"name": c.name, "arguments": dict(c.arguments)}}
            for c in message.tool_calls
        ]
    return [reply]
