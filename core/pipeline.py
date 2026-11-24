"""The request pipeline: composes the layers for one conversation turn.

At M1 it only holds the model layer. Later milestones insert guardrails, cache, router,
retrieval and the agent here, without changing the API above it.
"""

from collections.abc import AsyncIterator, Sequence

from core.config import ModelTier
from core.llm import Completion, LLMRequest, Message, ModelGateway, StreamEvent


class Pipeline:
    def __init__(
        self,
        gateway: ModelGateway,
        system_prompt: str,
        default_tier: ModelTier = ModelTier.MID,
    ) -> None:
        self._gateway = gateway
        self._system_prompt = system_prompt
        self._default_tier = default_tier

    async def answer(self, messages: Sequence[Message]) -> Completion:
        return await self._gateway.for_tier(self._default_tier).complete(self._request(messages))

    def stream(self, messages: Sequence[Message]) -> AsyncIterator[StreamEvent]:
        return self._gateway.for_tier(self._default_tier).stream(self._request(messages))

    def _request(self, messages: Sequence[Message]) -> LLMRequest:
        return LLMRequest(system=self._system_prompt, messages=tuple(messages))
