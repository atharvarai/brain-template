"""Adapter for a local Ollama server (free, offline). Base URL comes from OLLAMA_HOST."""

import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx2

from core.config import ModelSpec
from core.llm.errors import ProviderUnavailableError, error_for_status
from core.llm.types import Completion, LLMRequest, StopReason, StreamEvent, TextDelta, Usage

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
        return self._completion(data["message"]["content"], data, start)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        start = time.perf_counter()
        parts: list[str] = []
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
                    if text := data.get("message", {}).get("content"):
                        parts.append(text)
                        yield TextDelta(text)
                    if data.get("done"):
                        final = data
        except httpx2.HTTPError as exc:
            raise ProviderUnavailableError(f"{self.name}: {exc!r}") from exc
        yield self._completion("".join(parts), final, start)

    def _payload(self, request: LLMRequest, stream: bool) -> dict[str, Any]:
        structured = {"format": request.json_schema} if request.json_schema else {}
        return {
            **structured,
            "model": self._spec.model,
            "messages": [
                {"role": "system", "content": request.system},
                *({"role": m.role.value, "content": m.content} for m in request.messages),
            ],
            "stream": stream,
            "options": {"num_predict": request.max_tokens},
        }

    def _completion(self, text: str, data: dict[str, Any], start: float) -> Completion:
        return Completion(
            text=text,
            spec=self._spec,
            usage=Usage(int(data.get("prompt_eval_count", 0)), int(data.get("eval_count", 0))),
            stop_reason=_STOP_REASONS.get(str(data.get("done_reason")), StopReason.OTHER),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
