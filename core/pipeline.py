"""The request pipeline: composes the layers for one conversation turn.

So far: retrieve documents for the question, put them in front of it, and ask the model.
Later milestones insert guardrails, cache, router and the agent here, without changing the
API above it.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from core.config import ModelTier
from core.llm import Completion, LLMRequest, Message, ModelGateway, Role, TextDelta
from core.prompts import ComposedPrompt, Prompt, PromptRef
from core.rag import Chunk, Retriever, Source, format_documents, to_sources

# Until the access layer exists there is no user identity, so retrieval is limited to
# documents with no labels. Labelled (restricted) documents are indexed but never returned.
PUBLIC_ONLY: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class Answer:
    """A finished turn: the model's completion plus what it was grounded in."""

    completion: Completion
    sources: tuple[Source, ...]
    prompts: tuple[PromptRef, ...]


PipelineEvent = TextDelta | Answer


class Pipeline:
    def __init__(
        self,
        gateway: ModelGateway,
        system_prompt: ComposedPrompt,
        retriever: Retriever,
        turn_prompt: Prompt,
        top_k: int = 5,
        default_tier: ModelTier = ModelTier.MID,
    ) -> None:
        self._gateway = gateway
        self._system_prompt = system_prompt
        self._retriever = retriever
        self._turn_prompt = turn_prompt
        self._top_k = top_k
        self._default_tier = default_tier
        self._prompts = (*system_prompt.refs, turn_prompt.ref)

    async def answer(self, messages: Sequence[Message]) -> Answer:
        request, chunks = await self._prepare(messages)
        completion = await self._gateway.for_tier(self._default_tier).complete(request)
        return self._finish(completion, chunks)

    async def stream(self, messages: Sequence[Message]) -> AsyncIterator[PipelineEvent]:
        """Yield ``TextDelta`` events, then exactly one ``Answer``."""
        request, chunks = await self._prepare(messages)
        async for event in self._gateway.for_tier(self._default_tier).stream(request):
            if isinstance(event, TextDelta):
                yield event
            else:
                yield self._finish(event, chunks)

    async def _prepare(self, messages: Sequence[Message]) -> tuple[LLMRequest, list[Chunk]]:
        """Retrieve for the latest question and rewrite that turn to carry the documents.

        Documents go into the user turn, not the system prompt: the system prompt stays
        identical across requests, which is what makes it cacheable.
        """
        *history, last = messages
        hits = await self._retriever.search(last.content, self._top_k, PUBLIC_ONLY)
        chunks = [hit.chunk for hit in hits]
        turn = self._turn_prompt.render(documents=format_documents(chunks), question=last.content)
        request = LLMRequest(
            system=self._system_prompt.text, messages=(*history, Message(Role.USER, turn))
        )
        return request, chunks

    def _finish(self, completion: Completion, chunks: Sequence[Chunk]) -> Answer:
        return Answer(completion, to_sources(chunks, completion.text), self._prompts)
