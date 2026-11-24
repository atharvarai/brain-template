"""The request pipeline: composes the layers for one conversation turn.

So far it holds the prompt and model layers. Later milestones insert guardrails, cache,
router, retrieval and the agent here, without changing the API above it.
"""

from collections.abc import AsyncIterator, Sequence

from core.config import ModelTier
from core.llm import Completion, LLMRequest, Message, ModelGateway, StreamEvent
from core.prompts import ComposedPrompt, PromptRef


class Pipeline:
    def __init__(
        self,
        gateway: ModelGateway,
        system_prompt: ComposedPrompt,
        default_tier: ModelTier = ModelTier.MID,
    ) -> None:
        self._gateway = gateway
        self._system_prompt = system_prompt
        self._default_tier = default_tier

    @property
    def prompt_refs(self) -> tuple[PromptRef, ...]:
        """Which prompt files (name, version, checksum) every answer is built from."""
        return self._system_prompt.refs

    async def answer(self, messages: Sequence[Message]) -> Completion:
        return await self._gateway.for_tier(self._default_tier).complete(self._request(messages))

    def stream(self, messages: Sequence[Message]) -> AsyncIterator[StreamEvent]:
        return self._gateway.for_tier(self._default_tier).stream(self._request(messages))

    def _request(self, messages: Sequence[Message]) -> LLMRequest:
        return LLMRequest(system=self._system_prompt.text, messages=tuple(messages))
