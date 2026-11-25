"""The request pipeline: composes the layers for one conversation turn.

So far: retrieve documents for the question, put them in front of it, hand the turn to the
agent (model steps and tool calls within budgets), and trace every step. A run that wants
to change data pauses for approval and is resumed by a later request. Later milestones
insert guardrails, cache and router here, without changing the API above it.
"""

import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from core.access import (
    ANONYMOUS_CALLER,
    AuditAction,
    AuditEntry,
    AuditLog,
    Caller,
    InMemoryAuditLog,
    OpenPolicy,
    Policy,
)
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
from core.cache import AnswerCache, CacheHit, cache_scope
from core.config import ModelSpec, ModelTier, Provider
from core.guardrails import GuardFlag, Guardrails
from core.llm import (
    Completion,
    Message,
    ModelGateway,
    Role,
    StopReason,
    TextDelta,
    ToolCall,
    Usage,
)
from core.observability import ActiveTrace, Tracer
from core.prompts import ComposedPrompt, Prompt, PromptRef
from core.rag import Chunk, Retriever, Source, format_documents, to_sources
from core.router import FixedRouter, Router
from core.tools import ToolRegistry

RESUME_OPERATION = "resume"
NO_MODEL = ModelSpec(provider=Provider.NONE, model="none")  # for answers a guard gave


class AnswerStatus(StrEnum):
    COMPLETE = "complete"
    AWAITING_APPROVAL = "awaiting_approval"  # see ``pending``; resume with the ``run_id``
    BLOCKED = "blocked"  # a guardrail refused the message; no model was asked


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
    flags: tuple[GuardFlag, ...] = ()  # what the guardrails noticed
    replaced: bool = False  # a guardrail replaced the model's text (streamed text is void)
    tier: ModelTier | None = None  # the tier the router chose (None: no model was asked)
    cached: str | None = None  # "exact" or "semantic" when served from the answer cache


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
        guardrails: Guardrails | None = None,
        audit: AuditLog | None = None,
        router: Router | None = None,
        cache: AnswerCache | None = None,
        cache_context: Callable[[], dict[str, object]] = dict,
    ) -> None:
        """``cache_context`` returns what cached answers depend on besides the user's labels
        and the prompts: for example the model profile and the document index version."""
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
        self._guardrails = guardrails or Guardrails.off()
        self._audit: AuditLog = audit or InMemoryAuditLog()
        self._router: Router = router or FixedRouter(default_tier)
        self._cache = cache
        self._cache_context = cache_context
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
        use_cache: bool = True,
    ) -> Answer:
        return await _last(self._turn(messages, operation, caller, use_cache, stream=False))

    def stream(
        self,
        messages: Sequence[Message],
        *,
        operation: str = "chat",
        caller: Caller = ANONYMOUS_CALLER,
        use_cache: bool = True,
    ) -> AsyncIterator[PipelineEvent]:
        """Yield ``TextDelta`` and ``ToolActivity`` events, then exactly one ``Answer``."""
        return self._turn(messages, operation, caller, use_cache, stream=True)

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
            trace.set(tier=paused.progress.tier.value)
            events = self._agent.resume(
                self._gateway.for_tier(paused.progress.tier),
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
        self,
        messages: Sequence[Message],
        operation: str,
        caller: Caller,
        use_cache: bool,
        *,
        stream: bool,
    ) -> AsyncIterator[PipelineEvent]:
        async with self._tracer.trace(operation) as trace:
            trace.set(user=caller.user.id)
            check = await self._guardrails.check_input(messages[-1].content)
            if check.completion is not None:
                trace.record_completion(check.completion)
            self._note(check.flags, trace, caller)
            if check.blocked:
                yield self._blocked(check.message or "", check.flags, trace, messages[-1])
                return
            scope = self._scope(caller) if self._cache and use_cache else None
            hit = await self._cache.lookup(messages, scope) if self._cache and scope else None
            if hit is not None:
                answer = self._from_cache(hit, trace, messages[-1])
                if stream:
                    yield TextDelta(answer.completion.text)
                yield answer
                return
            decision = await self._router.route(messages)
            if decision.completion is not None:
                trace.record_completion(decision.completion)
            trace.set(tier=decision.tier.value, route=decision.reason[:120])
            turn, chunks, source_flags = await self._prepare(messages, trace, caller)
            progress = RunProgress(messages=list(turn), tier=decision.tier)
            noticed = (*check.flags, *source_flags)
            events = self._agent.run(
                self._gateway.for_tier(decision.tier),
                self._template(caller),
                progress,
                trace,
                stream=stream,
                caller=caller,
            )
            async for event in self._relay(events, progress, chunks, trace, caller, noticed):
                if isinstance(event, Answer) and scope is not None and _cacheable(event):
                    await self._remember(messages, scope, event)
                yield event

    async def _relay(
        self,
        events: AsyncIterator[AgentEvent],
        progress: RunProgress,
        chunks: Sequence[Chunk],
        trace: ActiveTrace,
        caller: Caller,
        noticed: tuple[GuardFlag, ...] = (),
    ) -> AsyncIterator[PipelineEvent]:
        """Pass text and tool events through; turn the agent's ending into an ``Answer``."""
        async for event in events:
            if isinstance(event, TextDelta | ToolActivity):
                yield event
            elif isinstance(event, Finished):
                yield self._finish(event.completion, progress, chunks, trace, caller, noticed)
            elif isinstance(event, ApprovalRequired):
                run_id = self._pause(progress, chunks, caller)
                trace.set(run_id=run_id)
                yield self._finish(
                    progress.completion(), progress, chunks, trace, caller, noticed,
                    run_id, event.pending,
                )  # fmt: skip

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
    ) -> tuple[list[Message], list[Chunk], tuple[GuardFlag, ...]]:
        """Retrieve for the latest question and rewrite that turn to carry the documents.

        Documents go into the user turn, not the system prompt: the system prompt stays
        identical across requests, which is what makes it cacheable.
        """
        *history, last = messages
        trace.record_prompts(str(ref) for ref in self._prompts)
        trace.capture(input=last.content)
        with trace.span("retrieve", k=self._top_k) as span:
            chunks, flags = self._guardrails.filter_sources(
                await self.retrieve(last.content, caller)
            )
            span["hits"] = len(chunks)
        self._note(flags, trace, caller)
        turn = self._turn_prompt.render(documents=format_documents(chunks), question=last.content)
        return [*history, Message(Role.USER, turn)], chunks, flags

    def _finish(
        self,
        completion: Completion,
        progress: RunProgress,
        chunks: Sequence[Chunk],
        trace: ActiveTrace,
        caller: Caller,
        noticed: tuple[GuardFlag, ...] = (),
        run_id: str | None = None,
        pending: tuple[ToolCall, ...] = (),
    ) -> Answer:
        # Everything the answer could be grounded in: the turns (documents, question,
        # earlier answers) and every tool result.
        evidence = [m.content for m in progress.messages]
        evidence += [r.content for m in progress.messages for r in m.tool_results]
        output = self._guardrails.check_output(completion.text, evidence)
        self._note(output.flags, trace, caller)
        completion = replace(completion, text=output.text)
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
            (*noticed, *output.flags),
            output.replaced,
            progress.tier,
        )

    def _scope(self, caller: Caller) -> str:
        """Who may share a cached answer: same readable labels, same prompts, same context."""
        return cache_scope(
            labels=sorted(self._policy.allowed_labels(caller.user)),
            prompts=[str(ref) for ref in self._prompts],
            **self._cache_context(),
        )

    def _from_cache(self, hit: CacheHit, trace: ActiveTrace, last: Message) -> Answer:
        body = hit.answer.body
        trace.event("cache_hit", kind=hit.kind, similarity=round(hit.similarity, 4))
        trace.capture(input=last.content, output=body["text"])
        spec = ModelSpec(provider=Provider(body["provider"]), model=body["model"])
        completion = Completion(body["text"], spec, Usage(), StopReason.END, 0)
        sources = tuple(Source(**source) for source in body["sources"])
        return Answer(
            completion, sources, self._prompts, trace.trace_id,
            tier=ModelTier(body["tier"]), cached=hit.kind,
        )  # fmt: skip

    async def _remember(self, messages: Sequence[Message], scope: str, answer: Answer) -> None:
        assert self._cache is not None and answer.tier is not None
        spec = answer.completion.spec
        body = {
            "text": answer.completion.text,
            "sources": [asdict(source) for source in answer.sources],
            "provider": spec.provider.value,
            "model": spec.model,
            "tier": answer.tier.value,
        }
        await self._cache.store(messages, scope, body)

    def _blocked(
        self, message: str, flags: tuple[GuardFlag, ...], trace: ActiveTrace, last: Message
    ) -> Answer:
        trace.capture(input=last.content, output=message)
        completion = Completion(message, NO_MODEL, Usage(), StopReason.END, 0)
        return Answer(
            completion, (), self._prompts, trace.trace_id,
            status=AnswerStatus.BLOCKED, flags=flags,
        )  # fmt: skip

    def _note(self, flags: Sequence[GuardFlag], trace: ActiveTrace, caller: Caller) -> None:
        """Every guard finding goes on the trace; blocks also go to the audit log."""
        for flag in flags:
            trace.event("guard", guard=flag.guard, blocked=flag.blocked, detail=flag.detail[:200])
            if flag.blocked:
                self._audit.record(
                    AuditEntry.of(
                        AuditAction.GUARD_BLOCKED, caller.user, flag.guard, "blocked",
                        trace_id=trace.trace_id, detail={"reason": flag.detail[:200]},
                    )
                )  # fmt: skip


def _cacheable(answer: Answer) -> bool:
    """Only answers that are the same for everyone with the same access: no tools ran (tool
    data is personal), no guardrail noticed anything, and the model finished normally."""
    return (
        answer.status is AnswerStatus.COMPLETE
        and not answer.tools
        and not answer.flags
        and not answer.replaced
        and answer.cached is None
        and answer.completion.stop_reason is StopReason.END
        and bool(answer.completion.text.strip())
    )


async def _last(events: AsyncIterator[PipelineEvent]) -> Answer:
    answer: Answer | None = None
    async for event in events:
        if isinstance(event, Answer):
            answer = event
    if answer is None:
        raise RuntimeError("the pipeline ended without an answer")
    return answer
