"""The request pipeline: composes the layers for one conversation turn.

So far: retrieve documents for the question, put them in front of it, hand the turn to the
agent (model steps and tool calls within budgets), and trace every step. A run that wants
to change data pauses for approval and is resumed by a later request. Later milestones
insert guardrails, cache and router here, without changing the API above it.
"""

import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from core.access import ANONYMOUS_CALLER, Caller, OpenPolicy, Policy
from core.agent import (
    Agent,
    AgentEvent,
    ApprovalRequired,
    Decision,
    Finished,
    InMemoryRunStore,
    PausedRun,
    RunNotFoundError,
    RunProgress,
    RunStore,
    StepTemplate,
    ToolActivity,
    validate_decisions,
)
from core.config import ModelTier
from core.llm import Completion, Message, ModelGateway, Role, TextDelta, ToolCall
from core.observability import ActiveTrace, Tracer
from core.prompts import ComposedPrompt, Prompt, PromptRef
from core.rag import Chunk, Retriever, Source, format_documents, to_sources
from core.tools import ToolRegistry

RESUME_OPERATION = "resume"


class AnswerStatus(StrEnum):
    COMPLETE = "complete"
    AWAITING_APPROVAL = "awaiting_approval"  # see ``pending``; resume with the ``run_id``


@dataclass(frozen=True, slots=True)
class Answer:
    """A finished (or paused) turn: the model's completion, what it was grounded in, the
    tools it ran, and its trace id.

    ``completion`` covers every model step of the run: text joined, usage added up.
    """

    completion: Completion
    sources: tuple[Source, ...]
    prompts: tuple[PromptRef, ...]
    trace_id: str
    tools: tuple[ToolActivity, ...] = ()
    status: AnswerStatus = AnswerStatus.COMPLETE
    run_id: str | None = None
    pending: tuple[ToolCall, ...] = ()  # actions waiting for approval, exactly as they will run


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
        agent: Agent | None = None,
        runs: RunStore | None = None,
        approval_ttl: timedelta = timedelta(hours=1),
        new_run_id: Callable[[], str] = lambda: uuid.uuid4().hex,
        policy: Policy | None = None,
    ) -> None:
        self._gateway = gateway
        self._system_prompt = system_prompt
        self._retriever = retriever
        self._turn_prompt = turn_prompt
        self._tracer = tracer
        self._top_k = top_k
        self._default_tier = default_tier
        self._agent = agent or Agent(ToolRegistry())
        self._runs: RunStore = runs or InMemoryRunStore()
        self._approval_ttl = approval_ttl
        self._new_run_id = new_run_id
        self._policy: Policy = policy or OpenPolicy()
        self._prompts = (*system_prompt.refs, turn_prompt.ref)

    async def retrieve(self, question: str, caller: Caller = ANONYMOUS_CALLER) -> list[Chunk]:
        """The documents a question would be answered from, limited to what this user may
        read: unlabelled documents plus the labels the policy gives their roles."""
        allowed = self._policy.allowed_labels(caller.user)
        hits = await self._retriever.search(question, self._top_k, allowed)
        return [hit.chunk for hit in hits]

    async def answer(
        self,
        messages: Sequence[Message],
        *,
        operation: str = "chat",
        caller: Caller = ANONYMOUS_CALLER,
    ) -> Answer:
        return await _last(self._turn(messages, operation, caller, stream=False))

    def stream(
        self,
        messages: Sequence[Message],
        *,
        operation: str = "chat",
        caller: Caller = ANONYMOUS_CALLER,
    ) -> AsyncIterator[PipelineEvent]:
        """Yield ``TextDelta`` and ``ToolActivity`` events, then exactly one ``Answer``."""
        return self._turn(messages, operation, caller, stream=True)

    async def resume(
        self, run_id: str, decisions: Sequence[Decision], caller: Caller = ANONYMOUS_CALLER
    ) -> Answer:
        """Apply a person's decisions to a paused run and carry it on to an answer."""
        return await _last(self.resume_stream(run_id, decisions, caller, stream=False))

    async def resume_stream(
        self,
        run_id: str,
        decisions: Sequence[Decision],
        caller: Caller = ANONYMOUS_CALLER,
        *,
        stream: bool = True,
    ) -> AsyncIterator[PipelineEvent]:
        # Check everything before claiming, so a bad request does not use up the approval.
        paused = self._runs.get(run_id)
        if paused.user_id != caller.user.id:  # someone else's run looks like no run at all
            raise RunNotFoundError(f"no paused run {run_id}")
        validate_decisions(paused.progress.pending, decisions)
        self._runs.claim(run_id)
        async with self._tracer.trace(RESUME_OPERATION) as trace:
            trace.set(resumes=run_id, user=caller.user.id)
            trace.record_prompts(str(ref) for ref in self._prompts)
            events = self._agent.resume(
                self._gateway.for_tier(self._default_tier),
                self._template(caller),
                paused.progress,
                decisions,
                trace,
                stream=stream,
                caller=caller,
            )
            async for event in self._relay(events, paused.progress, paused.chunks, trace, caller):
                yield event

    async def _turn(
        self, messages: Sequence[Message], operation: str, caller: Caller, *, stream: bool
    ) -> AsyncIterator[PipelineEvent]:
        async with self._tracer.trace(operation) as trace:
            trace.set(user=caller.user.id)
            turn, chunks = await self._prepare(messages, trace, caller)
            progress = RunProgress(messages=list(turn))
            events = self._agent.run(
                self._gateway.for_tier(self._default_tier),
                self._template(caller),
                progress,
                trace,
                stream=stream,
                caller=caller,
            )
            async for event in self._relay(events, progress, chunks, trace, caller):
                yield event

    async def _relay(
        self,
        events: AsyncIterator[AgentEvent],
        progress: RunProgress,
        chunks: Sequence[Chunk],
        trace: ActiveTrace,
        caller: Caller,
    ) -> AsyncIterator[PipelineEvent]:
        """Pass text and tool events through; turn the agent's ending into an ``Answer``."""
        async for event in events:
            if isinstance(event, TextDelta | ToolActivity):
                yield event
            elif isinstance(event, Finished):
                yield self._finish(event.completion, progress, chunks, trace)
            elif isinstance(event, ApprovalRequired):
                run_id = self._pause(progress, chunks, caller)
                trace.set(run_id=run_id)
                yield self._finish(
                    progress.completion(), progress, chunks, trace, run_id, event.pending
                )

    def _pause(self, progress: RunProgress, chunks: Sequence[Chunk], caller: Caller) -> str:
        now = datetime.now(UTC)
        run_id = self._new_run_id()
        expires = now + self._approval_ttl
        self._runs.save(PausedRun(run_id, now, expires, progress, tuple(chunks), caller.user.id))
        return run_id

    def _template(self, caller: Caller) -> StepTemplate:
        return StepTemplate(self._system_prompt.text, self._agent.tools_for(caller))

    async def _prepare(
        self, messages: Sequence[Message], trace: ActiveTrace, caller: Caller
    ) -> tuple[list[Message], list[Chunk]]:
        """Retrieve for the latest question and rewrite that turn to carry the documents.

        Documents go into the user turn, not the system prompt: the system prompt stays
        identical across requests, which is what makes it cacheable.
        """
        *history, last = messages
        trace.record_prompts(str(ref) for ref in self._prompts)
        trace.capture(input=last.content)
        with trace.span("retrieve", k=self._top_k) as span:
            chunks = await self.retrieve(last.content, caller)
            span["hits"] = len(chunks)
        turn = self._turn_prompt.render(documents=format_documents(chunks), question=last.content)
        return [*history, Message(Role.USER, turn)], chunks

    def _finish(
        self,
        completion: Completion,
        progress: RunProgress,
        chunks: Sequence[Chunk],
        trace: ActiveTrace,
        run_id: str | None = None,
        pending: tuple[ToolCall, ...] = (),
    ) -> Answer:
        sources = to_sources(chunks, completion.text)
        trace.record_retrieval(
            (chunk.id for chunk in chunks),
            (chunk.id for chunk, source in zip(chunks, sources, strict=True) if source.cited),
        )
        trace.capture(output=completion.text)
        trace.set(steps=progress.steps, tool_calls=progress.tool_calls)
        return Answer(
            completion,
            sources,
            self._prompts,
            trace.trace_id,
            tuple(progress.activity),
            AnswerStatus.AWAITING_APPROVAL if run_id else AnswerStatus.COMPLETE,
            run_id,
            pending,
        )


async def _last(events: AsyncIterator[PipelineEvent]) -> Answer:
    answer: Answer | None = None
    async for event in events:
        if isinstance(event, Answer):
            answer = event
    if answer is None:
        raise RuntimeError("the pipeline ended without an answer")
    return answer
