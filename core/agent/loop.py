"""The agent loop: model step, tool calls, results, repeat, within budgets.

Each step is one model call. If the model asks for tools, read-only ones run at once (and
concurrently); a call that changes data stops the run until a person approves or declines
it. When a budget is about to run out the model gets one last step with tools switched
off, so every run ends with an answer, never with a dangling tool call.

The loop keeps no state of its own: everything lives in ``RunProgress``, which is what
makes a paused run resumable from another request.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Collection, Mapping, Sequence

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
from core.agent.errors import DecisionError
from core.agent.types import (
    AgentEvent,
    ApprovalRequired,
    Budget,
    Decision,
    Finished,
    RunProgress,
    StepTemplate,
    ToolActivity,
)
from core.llm import (
    Completion,
    LLMProvider,
    LLMRequest,
    Message,
    Role,
    TextDelta,
    ToolCall,
    ToolChoice,
    ToolResult,
    ToolSpec,
)
from core.observability import ActiveTrace
from core.tools import ToolExecutor

DENIED = "Not run: you do not have permission to use this tool."
DECLINED = (
    "Not run: the user chose not to approve this action. Tell them plainly that it was not "
    "done because they declined it; it did not fail, so do not offer to retry it unasked."
)


class Agent:
    def __init__(
        self,
        tools: ToolExecutor,
        budget: Budget | None = None,
        auto_approve: Collection[str] = (),
        policy: Policy | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self._tools = tools
        self._budget = budget or Budget()
        self._auto_approve = frozenset(auto_approve)
        self._policy: Policy = policy or OpenPolicy()
        self._audit: AuditLog = audit or InMemoryAuditLog()

    @property
    def tools(self) -> ToolExecutor:
        return self._tools

    def tools_for(self, caller: Caller) -> tuple[ToolSpec, ...]:
        """The tools this user may call. The model is never shown the others."""
        return tuple(s for s in self._tools.specs if self._policy.can_call(caller.user, s.name))

    def needs_approval(self, name: str) -> bool:
        """Tools that may change data wait for a person. Unknown names never run anyway."""
        known = any(spec.name == name for spec in self._tools.specs)
        return known and not self._tools.is_read_only(name) and name not in self._auto_approve

    async def run(
        self,
        provider: LLMProvider,
        template: StepTemplate,
        progress: RunProgress,
        trace: ActiveTrace,
        *,
        stream: bool,
        caller: Caller = ANONYMOUS_CALLER,
    ) -> AsyncIterator[AgentEvent]:
        while True:
            last_step = not template.tools or self._exhausted(progress) is not None
            request = LLMRequest(
                system=template.system,
                messages=tuple(progress.messages),
                max_tokens=template.max_tokens,
                tools=template.tools,
                tool_choice=ToolChoice.NONE if last_step else ToolChoice.AUTO,
            )
            completion: Completion | None = None
            async for event in self._step(provider, request, progress, trace, stream=stream):
                if isinstance(event, TextDelta):
                    yield event
                else:
                    completion = event
            assert completion is not None
            if last_step or not completion.tool_calls:
                yield Finished(progress.completion())
                return

            calls = completion.tool_calls
            progress.messages.append(Message(Role.ASSISTANT, completion.text, tool_calls=calls))
            results, waiting, activity = await self._run_batch(calls, progress, trace, caller)
            for item in activity:
                yield item
            if waiting:
                progress.pending, progress.held = waiting, results
                progress.batch = tuple(call.id for call in calls)
                trace.event("approval_requested", tools=",".join(c.name for c in waiting))
                yield ApprovalRequired(waiting)
                return
            self._append_results(progress, [call.id for call in calls], results, trace)

    async def resume(
        self,
        provider: LLMProvider,
        template: StepTemplate,
        progress: RunProgress,
        decisions: Sequence[Decision],
        trace: ActiveTrace,
        *,
        stream: bool,
        caller: Caller = ANONYMOUS_CALLER,
    ) -> AsyncIterator[AgentEvent]:
        """Act on the decisions for a paused run, then carry on with the loop."""
        by_id = validate_decisions(progress.pending, decisions)
        approved = [call for call in progress.pending if by_id[call.id].approve]
        for call in progress.pending:
            decision = by_id[call.id]
            trace.event("approval_decided", tool=call.name, approved=decision.approve)
            outcome = "approved" if decision.approve else "declined"
            self._audit.record(
                AuditEntry.of(
                    AuditAction.APPROVAL, caller.user, call.name, outcome,
                    trace_id=trace.trace_id, detail={"arguments": dict(call.arguments)},
                )
            )  # fmt: skip
            if not decision.approve:
                note = f" Their note: {decision.note}" if decision.note else ""
                progress.held[call.id] = ToolResult(call.id, call.name, DECLINED + note, True)
        results, activity = await self._execute(approved, progress, trace, caller)
        progress.held.update(results)
        for item in activity:
            yield item
        batch, held = list(progress.batch), progress.held
        progress.pending, progress.held, progress.batch = (), {}, ()
        self._append_results(progress, batch, held, trace)
        async for event in self.run(
            provider, template, progress, trace, stream=stream, caller=caller
        ):
            yield event

    # --- one step -------------------------------------------------------------------------

    async def _step(
        self,
        provider: LLMProvider,
        request: LLMRequest,
        progress: RunProgress,
        trace: ActiveTrace,
        *,
        stream: bool,
    ) -> AsyncIterator[TextDelta | Completion]:
        step = progress.steps + 1
        attributes = {"step": step, "tools_offered": request.offers_tools}
        with trace.span("generate", **attributes) as span:
            if stream:
                completion = None
                first = True
                async for event in provider.stream(request):
                    if isinstance(event, TextDelta):
                        if first and progress.texts:  # keep steps apart in the joined text
                            yield TextDelta("\n\n")
                        if first and step == 1:
                            trace.event("first_token")
                        first = False
                        yield event
                    else:
                        completion = event
                if completion is None:
                    raise RuntimeError(f"{provider.name} ended its stream without a completion")
            else:
                completion = await provider.complete(request)
            span["model"] = completion.spec.model
            span["tool_calls"] = len(completion.tool_calls)
        trace.record_completion(completion)
        progress.record(completion)
        yield completion

    # --- tools ----------------------------------------------------------------------------

    async def _run_batch(
        self,
        calls: Sequence[ToolCall],
        progress: RunProgress,
        trace: ActiveTrace,
        caller: Caller,
    ) -> tuple[dict[str, ToolResult], tuple[ToolCall, ...], list[ToolActivity]]:
        """Run what may run now. Returns the results so far, the calls waiting for approval
        and the tool runs that happened.

        Calls beyond the tool-call budget get an error result instead of running: every
        call the model made needs a result.
        """
        room = max(self._budget.max_tool_calls - progress.tool_calls, 0)
        allowed, over = calls[:room], calls[room:]
        results = {
            call.id: ToolResult(
                call.id, call.name,
                f"Not run: the limit of {self._budget.max_tool_calls} tool calls for this "
                "question is reached.", is_error=True,
            )
            for call in over
        }  # fmt: skip
        permitted = []
        for call in allowed:  # checked again here: the model may name a tool it was not shown
            if self._policy.can_call(caller.user, call.name):
                permitted.append(call)
                continue
            results[call.id] = ToolResult(call.id, call.name, DENIED, is_error=True)
            trace.event("tool_denied", tool=call.name)
            self._audit.record(
                AuditEntry.of(AuditAction.TOOL_DENIED, caller.user, call.name, "denied",
                              trace_id=trace.trace_id)
            )  # fmt: skip
        waiting = tuple(call for call in permitted if self.needs_approval(call.name))
        now = [call for call in permitted if call not in waiting]
        progress.tool_calls += len(allowed)
        ran, activity = await self._execute(now, progress, trace, caller)
        results.update(ran)
        return results, waiting, activity

    async def _execute(
        self,
        calls: Sequence[ToolCall],
        progress: RunProgress,
        trace: ActiveTrace,
        caller: Caller,
    ) -> tuple[dict[str, ToolResult], list[ToolActivity]]:
        step = progress.steps

        async def run(call: ToolCall) -> tuple[ToolResult, ToolActivity]:
            start = time.perf_counter()
            with trace.span("tool", tool=call.name, step=step) as span:
                result = await self._tools.run(call, caller.token)
                span["ok"] = not result.is_error
            self._audit.record(
                AuditEntry.of(
                    AuditAction.TOOL_CALL, caller.user, call.name,
                    "error" if result.is_error else "ok",
                    trace_id=trace.trace_id, detail={"arguments": dict(call.arguments)},
                )
            )  # fmt: skip
            elapsed = round((time.perf_counter() - start) * 1000)
            return result, ToolActivity(call.name, not result.is_error, elapsed, step)

        ran = await asyncio.gather(*(run(call) for call in calls))
        activity = [item for _, item in ran]
        progress.activity.extend(activity)
        return {result.call_id: result for result, _ in ran}, activity

    def _append_results(
        self,
        progress: RunProgress,
        order: Sequence[str],
        results: Mapping[str, ToolResult],
        trace: ActiveTrace,
    ) -> None:
        reason = self._exhausted(progress)
        note = ""
        if reason is not None:
            trace.event("budget_reached", reason=reason)
            note = f"(No more tool calls for this question: {reason}. Answer from what you have.)"
        ordered = tuple(results[call_id] for call_id in order)
        progress.messages.append(Message(Role.USER, note, tool_results=ordered))

    def _exhausted(self, progress: RunProgress) -> str | None:
        """Why the next step must be the last one, if it must."""
        budget = self._budget
        if progress.steps >= budget.max_steps - 1:
            return f"step limit of {budget.max_steps} reached"
        if progress.tokens >= budget.max_tokens:
            return f"token budget of {budget.max_tokens} used"
        if progress.tool_calls >= budget.max_tool_calls:
            return f"tool-call limit of {budget.max_tool_calls} reached"
        return None


def validate_decisions(
    pending: Sequence[ToolCall], decisions: Sequence[Decision]
) -> dict[str, Decision]:
    """Every waiting call needs exactly one decision, and nothing else may be decided."""
    by_id: dict[str, Decision] = {}
    for decision in decisions:
        if decision.call_id in by_id:
            raise DecisionError(f"call {decision.call_id} is decided twice")
        by_id[decision.call_id] = decision
    waiting = {call.id for call in pending}
    if unknown := sorted(by_id.keys() - waiting):
        raise DecisionError(f"no action waiting with call id(s) {unknown}")
    if missing := sorted(waiting - by_id.keys()):
        raise DecisionError(f"no decision for call id(s) {missing}")
    return by_id
