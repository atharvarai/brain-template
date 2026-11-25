"""The request pipeline: composes the layers for one conversation turn.

So far: retrieve documents for the question, put them in front of it, offer the company's
tools, ask the model, run the tools it asks for (one round), ask again for the final answer,
and trace every step. Later milestones insert guardrails, cache, router and the multi-step
agent here, without changing the API above it.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace

from core.config import ModelTier
from core.llm import (
    Completion,
    LLMProvider,
    LLMRequest,
    Message,
    ModelGateway,
    Role,
    TextDelta,
    ToolCall,
    ToolChoice,
    ToolResult,
    Usage,
)
from core.observability import ActiveTrace, Tracer
from core.prompts import ComposedPrompt, Prompt, PromptRef
from core.rag import Chunk, Retriever, Source, format_documents, to_sources
from core.tools import ToolExecutor, ToolRegistry

# Until the access layer exists there is no user identity, so retrieval is limited to
# documents with no labels. Labelled (restricted) documents are indexed but never returned.
PUBLIC_ONLY: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ToolActivity:
    """One tool run during a turn. Streamed as it finishes, and kept on the answer."""

    name: str
    ok: bool
    latency_ms: int


@dataclass(frozen=True, slots=True)
class Answer:
    """A finished turn: the model's completion, what it was grounded in, and its trace id.

    When tools ran, ``completion`` covers both model calls: text joined, usage added up.
    """

    completion: Completion
    sources: tuple[Source, ...]
    prompts: tuple[PromptRef, ...]
    trace_id: str
    tools: tuple[ToolActivity, ...] = ()


PipelineEvent = TextDelta | ToolActivity | Answer


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
        tools: ToolExecutor | None = None,
        max_tool_calls: int = 4,
    ) -> None:
        self._gateway = gateway
        self._system_prompt = system_prompt
        self._retriever = retriever
        self._turn_prompt = turn_prompt
        self._tracer = tracer
        self._top_k = top_k
        self._default_tier = default_tier
        self._tools: ToolExecutor = tools or ToolRegistry()
        self._max_tool_calls = max_tool_calls
        self._prompts = (*system_prompt.refs, turn_prompt.ref)

    async def retrieve(self, question: str) -> list[Chunk]:
        """The documents a question would be answered from. Also used by retrieval evals."""
        hits = await self._retriever.search(question, self._top_k, PUBLIC_ONLY)
        return [hit.chunk for hit in hits]

    async def answer(self, messages: Sequence[Message], *, operation: str = "chat") -> Answer:
        async with self._tracer.trace(operation) as trace:
            request, chunks = await self._prepare(messages, trace)
            provider = self._gateway.for_tier(self._default_tier)
            completion = await self._complete(provider, request, trace)
            activity: tuple[ToolActivity, ...] = ()
            if completion.tool_calls:
                results, activity = await self._run_tools(completion.tool_calls, trace)
                follow_up = _follow_up(request, completion, results)
                final = await self._complete(provider, follow_up, trace)
                completion = _merge(completion, final)
            return self._finish(completion, chunks, activity, trace)

    async def stream(
        self, messages: Sequence[Message], *, operation: str = "chat"
    ) -> AsyncIterator[PipelineEvent]:
        """Yield ``TextDelta`` and ``ToolActivity`` events, then exactly one ``Answer``."""
        async with self._tracer.trace(operation) as trace:
            request, chunks = await self._prepare(messages, trace)
            provider = self._gateway.for_tier(self._default_tier)
            completion: Completion | None = None
            async for event in self._stream_once(provider, request, trace):
                if isinstance(event, TextDelta):
                    yield event
                else:
                    completion = event
            assert completion is not None  # _stream_once always ends with one
            activity: tuple[ToolActivity, ...] = ()
            if completion.tool_calls:
                results, activity = await self._run_tools(completion.tool_calls, trace)
                for item in activity:
                    yield item
                if completion.text:
                    yield TextDelta("\n\n")
                final: Completion | None = None
                async for event in self._stream_once(
                    provider, _follow_up(request, completion, results), trace
                ):
                    if isinstance(event, TextDelta):
                        yield event
                    else:
                        final = event
                assert final is not None
                completion = _merge(completion, final)
            yield self._finish(completion, chunks, activity, trace)

    async def _complete(
        self, provider: LLMProvider, request: LLMRequest, trace: ActiveTrace
    ) -> Completion:
        with trace.span("generate", tier=self._default_tier.value) as span:
            completion = await provider.complete(request)
            span["model"] = completion.spec.model
            span["tool_calls"] = len(completion.tool_calls)
        trace.record_completion(completion)
        return completion

    async def _stream_once(
        self, provider: LLMProvider, request: LLMRequest, trace: ActiveTrace
    ) -> AsyncIterator[TextDelta | Completion]:
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
                    span["tool_calls"] = len(event.tool_calls)
                    completion = event
        if completion is None:
            raise RuntimeError(f"{provider.name} ended its stream without a completion")
        trace.record_completion(completion)
        yield completion

    async def _run_tools(
        self, calls: Sequence[ToolCall], trace: ActiveTrace
    ) -> tuple[tuple[ToolResult, ...], tuple[ToolActivity, ...]]:
        """Run the calls concurrently. Calls over the limit are answered with an error, since
        every call the model made needs a result."""
        allowed, skipped = calls[: self._max_tool_calls], calls[self._max_tool_calls :]

        async def run(call: ToolCall) -> tuple[ToolResult, ToolActivity]:
            start = time.perf_counter()
            with trace.span("tool", tool=call.name) as span:
                result = await self._tools.run(call)
                span["ok"] = not result.is_error
            elapsed = round((time.perf_counter() - start) * 1000)
            return result, ToolActivity(call.name, not result.is_error, elapsed)

        ran = await asyncio.gather(*(run(call) for call in allowed))
        limit = f"Not run: at most {self._max_tool_calls} tool calls are allowed per turn."
        results = [result for result, _ in ran]
        results += [ToolResult(c.id, c.name, limit, is_error=True) for c in skipped]
        return tuple(results), tuple(activity for _, activity in ran)

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
            system=self._system_prompt.text,
            messages=(*history, Message(Role.USER, turn)),
            tools=self._tools.specs,
        )
        return request, chunks

    def _finish(
        self,
        completion: Completion,
        chunks: Sequence[Chunk],
        activity: tuple[ToolActivity, ...],
        trace: ActiveTrace,
    ) -> Answer:
        sources = to_sources(chunks, completion.text)
        trace.record_retrieval(
            (chunk.id for chunk in chunks),
            (chunk.id for chunk, source in zip(chunks, sources, strict=True) if source.cited),
        )
        trace.capture(output=completion.text)
        return Answer(completion, sources, self._prompts, trace.trace_id, activity)


def _follow_up(
    request: LLMRequest, completion: Completion, results: Sequence[ToolResult]
) -> LLMRequest:
    """The second model call: the tool calls and their results appended, and no further
    calls allowed. A multi-step loop with budgets is the agent's job (layer 05)."""
    return replace(
        request,
        messages=(
            *request.messages,
            Message(Role.ASSISTANT, completion.text, tool_calls=completion.tool_calls),
            Message(Role.USER, "", tool_results=tuple(results)),
        ),
        tool_choice=ToolChoice.NONE,
    )


def _merge(first: Completion, final: Completion) -> Completion:
    """One completion for the whole turn: text joined, usage and latency added up."""
    text = "\n\n".join(part for part in (first.text, final.text) if part)
    usage = Usage(
        first.usage.input_tokens + final.usage.input_tokens,
        first.usage.output_tokens + final.usage.output_tokens,
    )
    return replace(final, text=text, usage=usage, latency_ms=first.latency_ms + final.latency_ms)
