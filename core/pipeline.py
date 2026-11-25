"""The request pipeline: composes the layers for one conversation turn.

So far: retrieve documents for the question, put them in front of it, ask the model, and
trace every step. Later milestones insert guardrails, cache, router and the agent here,
without changing the API above it.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from core.config import ModelTier
from core.llm import Completion, LLMRequest, Message, ModelGateway, Role, TextDelta
from core.observability import ActiveTrace, Tracer
from core.prompts import ComposedPrompt, Prompt, PromptRef
from core.rag import Chunk, Retriever, Source, format_documents, to_sources

# Until the access layer exists there is no user identity, so retrieval is limited to
# documents with no labels. Labelled (restricted) documents are indexed but never returned.
PUBLIC_ONLY: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class Answer:
    """A finished turn: the model's completion, what it was grounded in, and its trace id."""

    completion: Completion
    sources: tuple[Source, ...]
    prompts: tuple[PromptRef, ...]
    trace_id: str


PipelineEvent = TextDelta | Answer


class Pipeline:
    def __init__(
        self,
        gateway: ModelGateway,
        system_prompt: ComposedPrompt,
        retriever: Retriever,
        turn_prompt: Prompt,
        tracer: Tracer,
        top_k: int = 5,
        default_tier: ModelTier = ModelTier.MID,
    ) -> None:
        self._gateway = gateway
        self._system_prompt = system_prompt
        self._retriever = retriever
        self._turn_prompt = turn_prompt
        self._tracer = tracer
        self._top_k = top_k
        self._default_tier = default_tier
        self._prompts = (*system_prompt.refs, turn_prompt.ref)

    async def retrieve(self, question: str) -> list[Chunk]:
        """The documents a question would be answered from. Also used by retrieval evals."""
        hits = await self._retriever.search(question, self._top_k, PUBLIC_ONLY)
        return [hit.chunk for hit in hits]

    async def answer(self, messages: Sequence[Message], *, operation: str = "chat") -> Answer:
        async with self._tracer.trace(operation) as trace:
            request, chunks = await self._prepare(messages, trace)
            with trace.span("generate", tier=self._default_tier.value) as span:
                completion = await self._gateway.for_tier(self._default_tier).complete(request)
                span["model"] = completion.spec.model
            return self._finish(completion, chunks, trace)

    async def stream(
        self, messages: Sequence[Message], *, operation: str = "chat"
    ) -> AsyncIterator[PipelineEvent]:
        """Yield ``TextDelta`` events, then exactly one ``Answer``."""
        async with self._tracer.trace(operation) as trace:
            request, chunks = await self._prepare(messages, trace)
            provider = self._gateway.for_tier(self._default_tier)
            first = True
            completion: Completion | None = None
            with trace.span("generate", tier=self._default_tier.value) as span:
                async for event in provider.stream(request):
                    if isinstance(event, TextDelta):
                        if first:
                            trace.event("first_token")
                            first = False
                        yield event
                    else:
                        span["model"] = event.spec.model
                        completion = event
            if completion is None:
                raise RuntimeError(f"{provider.name} ended its stream without a completion")
            yield self._finish(completion, chunks, trace)

    async def _prepare(
        self, messages: Sequence[Message], trace: ActiveTrace
    ) -> tuple[LLMRequest, list[Chunk]]:
        """Retrieve for the latest question and rewrite that turn to carry the documents.

        Documents go into the user turn, not the system prompt: the system prompt stays
        identical across requests, which is what makes it cacheable.
        """
        *history, last = messages
        trace.record_prompts(str(ref) for ref in self._prompts)
        trace.capture(input=last.content)
        with trace.span("retrieve", k=self._top_k) as span:
            chunks = await self.retrieve(last.content)
            span["hits"] = len(chunks)
        turn = self._turn_prompt.render(documents=format_documents(chunks), question=last.content)
        request = LLMRequest(
            system=self._system_prompt.text, messages=(*history, Message(Role.USER, turn))
        )
        return request, chunks

    def _finish(
        self, completion: Completion, chunks: Sequence[Chunk], trace: ActiveTrace
    ) -> Answer:
        sources = to_sources(chunks, completion.text)
        trace.record_completion(completion)
        trace.record_retrieval(
            (chunk.id for chunk in chunks),
            (chunk.id for chunk, source in zip(chunks, sources, strict=True) if source.cited),
        )
        trace.capture(output=completion.text)
        return Answer(completion, sources, self._prompts, trace.trace_id)
