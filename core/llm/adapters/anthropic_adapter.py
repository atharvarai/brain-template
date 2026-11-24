"""Adapter for the Anthropic Messages API. Reads ANTHROPIC_API_KEY from the environment."""

import time
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic.types import Message as AnthropicMessage
from anthropic.types import MessageParam

from core.config import ModelSpec
from core.llm.errors import ProviderUnavailableError, error_for_status
from core.llm.types import (
    Completion,
    LLMRequest,
    Role,
    StopReason,
    StreamEvent,
    TextDelta,
    Usage,
)

_STOP_REASONS = {
    "end_turn": StopReason.END,
    "stop_sequence": StopReason.END,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
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
        )


def _structured_output(request: LLMRequest) -> dict[str, Any]:
    if request.json_schema is None:
        return {}
    return {"output_config": {"format": {"type": "json_schema", "schema": request.json_schema}}}


def _to_params(request: LLMRequest) -> list[MessageParam]:
    params: list[MessageParam] = []
    for message in request.messages:
        if message.role is Role.USER:
            params.append({"role": "user", "content": message.content})
        else:
            params.append({"role": "assistant", "content": message.content})
    return params
