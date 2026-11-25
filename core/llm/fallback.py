"""Failover across providers, with a circuit breaker per provider.

``FallbackChain`` is itself an ``LLMProvider`` (Composite), so callers cannot tell one
provider from several.
"""

import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field

from core.llm.errors import AllProvidersFailedError, ProviderUnavailableError
from core.llm.provider import LLMProvider
from core.llm.types import Completion, LLMRequest, StreamEvent
from core.observability import record_event

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreaker:
    """Stops calling a failing provider for a while, then lets one trial call through.

    closed (normal) -> open after ``failure_threshold`` consecutive failures ->
    half-open once ``cooldown_seconds`` have passed -> closed again on the next success.
    """

    failure_threshold: int = 3
    cooldown_seconds: float = 30.0
    clock: Callable[[], float] = time.monotonic
    _failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)

    def allows_request(self) -> bool:
        if self._failures < self.failure_threshold:
            return True
        return self.clock() - self._opened_at >= self.cooldown_seconds

    def record_success(self) -> None:
        self._failures = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = self.clock()


class FallbackChain:
    """Tries providers in order and returns the first answer."""

    def __init__(
        self,
        providers: Sequence[LLMProvider],
        breaker_factory: Callable[[], CircuitBreaker] = CircuitBreaker,
    ) -> None:
        if not providers:
            raise ValueError("a fallback chain needs at least one provider")
        self._links = [(provider, breaker_factory()) for provider in providers]

    @property
    def name(self) -> str:
        return "fallback(" + ", ".join(provider.name for provider, _ in self._links) + ")"

    async def complete(self, request: LLMRequest) -> Completion:
        errors: list[str] = []
        for provider, breaker in self._links:
            if not breaker.allows_request():
                errors.append(f"{provider.name}: circuit open")
                record_event("provider_skipped", provider=provider.name, reason="circuit open")
                continue
            try:
                completion = await provider.complete(request)
            except ProviderUnavailableError as exc:
                self._failed(provider, breaker, exc, errors)
                continue
            breaker.record_success()
            return completion
        raise AllProvidersFailedError(tuple(errors))

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Fail over only while nothing has been sent; a half-streamed answer is not retried."""
        errors: list[str] = []
        for provider, breaker in self._links:
            if not breaker.allows_request():
                errors.append(f"{provider.name}: circuit open")
                record_event("provider_skipped", provider=provider.name, reason="circuit open")
                continue
            started = False
            try:
                async for event in provider.stream(request):
                    started = True
                    yield event
            except ProviderUnavailableError as exc:
                if started:
                    breaker.record_failure()
                    record_event("provider_failed_mid_stream", provider=provider.name)
                    raise
                self._failed(provider, breaker, exc, errors)
                continue
            breaker.record_success()
            return
        raise AllProvidersFailedError(tuple(errors))

    @staticmethod
    def _failed(
        provider: LLMProvider,
        breaker: CircuitBreaker,
        exc: ProviderUnavailableError,
        errors: list[str],
    ) -> None:
        breaker.record_failure()
        errors.append(str(exc))
        logger.warning("provider unavailable, trying next: %s", exc)
        record_event("provider_failed", provider=provider.name, error=str(exc)[:200])
