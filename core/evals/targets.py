"""What an eval runs against. Strategy: the full answer path, or retrieval alone."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from core.access import ANONYMOUS_CALLER, Caller
from core.agent import Decision
from core.config import Provider
from core.evals.cases import EvalCase
from core.observability import PriceBook
from core.pipeline import AnswerStatus, Pipeline

EVAL_OPERATION = "eval"
MAX_RESUMES = 5  # a run that keeps pausing is cut off here


class EvalMode(StrEnum):
    ANSWER = "answer"  # ask the model: every check runs (needs a working model)
    RETRIEVAL = "retrieval"  # retrieval only: offline, free, fast enough for CI


@dataclass(frozen=True, slots=True)
class EvalOutcome:
    """What the brain produced for one case."""

    answer: str
    retrieved: tuple[str, ...]  # "source#heading" of each chunk given to the model, in order
    cited: tuple[str, ...]  # the same, for chunks the answer cited
    latency_ms: int
    trace_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    tools_called: tuple[str, ...] = ()  # in the order they ran
    approvals_requested: tuple[str, ...] = ()  # tools the brain paused on
    blocked: bool = False  # a guardrail refused the message
    guard_flags: tuple[str, ...] = ()  # "guard: detail" for everything the guardrails noticed


class EvalTarget(Protocol):
    mode: EvalMode

    async def run(self, case: EvalCase) -> EvalOutcome: ...


CallerFor = Callable[[EvalCase], Caller]


def _anonymous(_: EvalCase) -> Caller:
    return ANONYMOUS_CALLER


class AnswerTarget:
    mode = EvalMode.ANSWER

    def __init__(
        self, pipeline: Pipeline, prices: PriceBook, caller_for: CallerFor = _anonymous
    ) -> None:
        self._pipeline = pipeline
        self._prices = prices
        self._caller_for = caller_for

    async def run(self, case: EvalCase) -> EvalOutcome:
        start = time.perf_counter()
        caller = self._caller_for(case)
        answer = await self._pipeline.answer(
            case.messages(), operation=EVAL_OPERATION, caller=caller
        )
        asked: list[str] = []
        for _ in range(MAX_RESUMES):
            if answer.run_id is None:
                break
            asked += [call.name for call in answer.pending]
            decisions = [Decision(call.id, case.approve) for call in answer.pending]
            answer = await self._pipeline.resume(answer.run_id, decisions, caller)
        usage, spec = answer.completion.usage, answer.completion.spec
        return EvalOutcome(
            answer=answer.completion.text,
            retrieved=tuple(f"{s.source}#{s.section}" for s in answer.sources),
            cited=tuple(f"{s.source}#{s.section}" for s in answer.sources if s.cited),
            latency_ms=round((time.perf_counter() - start) * 1000),
            trace_id=answer.trace_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=(
                0.0  # a guardrail answered: no model was called
                if spec.provider is Provider.NONE
                else self._prices.cost(spec.model, usage.input_tokens, usage.output_tokens)
            ),
            tools_called=tuple(activity.name for activity in answer.tools),
            approvals_requested=tuple(asked),
            blocked=answer.status is AnswerStatus.BLOCKED,
            guard_flags=tuple(f"{f.guard}: {f.detail}" for f in answer.flags),
        )


class RetrievalTarget:
    mode = EvalMode.RETRIEVAL

    def __init__(self, pipeline: Pipeline, caller_for: CallerFor = _anonymous) -> None:
        self._pipeline = pipeline
        self._caller_for = caller_for

    async def run(self, case: EvalCase) -> EvalOutcome:
        start = time.perf_counter()
        chunks = await self._pipeline.retrieve(case.question, self._caller_for(case))
        return EvalOutcome(
            answer="",
            retrieved=tuple(f"{chunk.source}#{chunk.heading}" for chunk in chunks),
            cited=(),
            latency_ms=round((time.perf_counter() - start) * 1000),
        )
