"""Adapter for the OpenAI Chat Completions API. Reads OPENAI_API_KEY from the environment."""

import time
from collections.abc import AsyncIterator

import openai
from openai.types.chat import ChatCompletionMessageParam

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
    "stop": StopReason.END,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
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
            )
        except openai.APIConnectionError as exc:  # includes timeouts
            raise ProviderUnavailableError(f"{self.name}: {exc}") from exc
        except openai.APIStatusError as exc:
            raise error_for_status(self.name, exc.status_code, exc.message) from exc

        choice = response.choices[0]
        usage = response.usage
        return self._completion(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=Usage(usage.prompt_tokens, usage.completion_tokens) if usage else Usage(),
            start=start,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        start = time.perf_counter()
        parts: list[str] = []
        finish_reason: str | None = None
        usage = Usage()
        try:
            chunks = await self._client.chat.completions.create(
                model=self._spec.model,
                messages=_to_params(request),
                max_completion_tokens=request.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in chunks:
                if chunk.usage:  # arrives on the last chunk, which has no choices
                    usage = Usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                for choice in chunk.choices:
                    if choice.delta.content:
                        parts.append(choice.delta.content)
                        yield TextDelta(choice.delta.content)
                    finish_reason = choice.finish_reason or finish_reason
        except openai.APIConnectionError as exc:
            raise ProviderUnavailableError(f"{self.name}: {exc}") from exc
        except openai.APIStatusError as exc:
            raise error_for_status(self.name, exc.status_code, exc.message) from exc
        yield self._completion("".join(parts), finish_reason, usage, start)

    def _completion(
        self, text: str, finish_reason: str | None, usage: Usage, start: float
    ) -> Completion:
        return Completion(
            text=text,
            spec=self._spec,
            usage=usage,
            stop_reason=_STOP_REASONS.get(str(finish_reason), StopReason.OTHER),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )


def _to_params(request: LLMRequest) -> list[ChatCompletionMessageParam]:
    params: list[ChatCompletionMessageParam] = [{"role": "system", "content": request.system}]
    for message in request.messages:
        if message.role is Role.USER:
            params.append({"role": "user", "content": message.content})
        else:
            params.append({"role": "assistant", "content": message.content})
    return params
