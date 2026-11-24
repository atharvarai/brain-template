"""The port every model adapter implements."""

from collections.abc import AsyncIterator
from typing import Protocol

from core.llm.types import Completion, LLMRequest, StreamEvent


class LLMProvider(Protocol):
    """Something that can answer an ``LLMRequest``.

    Implementations raise ``ProviderUnavailableError`` for failures another provider could
    survive, and ``ProviderRequestError`` for requests that are wrong in themselves.
    """

    @property
    def name(self) -> str:
        """Short label for logs and traces, for example ``anthropic:claude-opus-5-5``."""
        ...

    async def complete(self, request: LLMRequest) -> Completion:
        """Return the whole answer at once."""
        ...

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Yield ``TextDelta`` events, then exactly one final ``Completion``."""
        ...
