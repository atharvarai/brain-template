"""Runs eval cases against a target, a few at a time, and collects the results."""

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from core.evals.cases import EvalCase
from core.evals.checks import Check, CheckKind, CheckResult, applicable, build_checks
from core.evals.judge import Judge
from core.evals.targets import EvalMode, EvalOutcome, EvalTarget
from core.observability import percentile

logger = logging.getLogger(__name__)

_KINDS = {
    EvalMode.ANSWER: frozenset(CheckKind),
    EvalMode.RETRIEVAL: frozenset({CheckKind.RETRIEVAL}),
}


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    tags: tuple[str, ...]
    passed: bool
    checks: tuple[CheckResult, ...]
    outcome: EvalOutcome | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EvalReport:
    pack: str
    mode: EvalMode
    profile: str
    started_at: datetime
    duration_ms: int
    results: tuple[CaseResult, ...]
    skipped: tuple[str, ...] = field(default=())  # cases with no check for this mode

    @property
    def passed(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed / len(self.results) if self.results else 0.0

    def by_tag(self) -> dict[str, tuple[int, int]]:
        """tag -> (passed, total)"""
        totals: dict[str, tuple[int, int]] = {}
        for result in self.results:
            for tag in result.tags:
                passed, total = totals.get(tag, (0, 0))
                totals[tag] = (passed + result.passed, total + 1)
        return dict(sorted(totals.items()))

    def by_check(self) -> dict[str, tuple[int, int]]:
        """check name -> (passed, total)"""
        totals: dict[str, tuple[int, int]] = {}
        for result in self.results:
            for check in result.checks:
                passed, total = totals.get(check.name, (0, 0))
                totals[check.name] = (passed + check.passed, total + 1)
        return totals

    def latency_ms(self, fraction: float) -> int:
        return percentile([r.outcome.latency_ms for r in self.results if r.outcome], fraction)

    @property
    def tokens(self) -> tuple[int, int]:
        outcomes = [r.outcome for r in self.results if r.outcome]
        return sum(o.input_tokens for o in outcomes), sum(o.output_tokens for o in outcomes)

    @property
    def cost_usd(self) -> float | None:
        costs = [r.outcome.cost_usd for r in self.results if r.outcome]
        if not costs or any(cost is None for cost in costs):
            return None
        return sum(cost for cost in costs if cost is not None)


class EvalRunner:
    def __init__(
        self,
        target: EvalTarget,
        judge: Judge | None,
        pack: str,
        profile: str,
        concurrency: int = 4,
    ) -> None:
        self._target = target
        self._judge = judge
        self._pack = pack
        self._profile = profile
        self._limit = asyncio.Semaphore(concurrency)

    async def run(self, cases: Sequence[EvalCase]) -> EvalReport:
        kinds = _KINDS[self._target.mode]
        planned = [
            (case, applicable(build_checks(case.expect, self._judge), kinds)) for case in cases
        ]
        runnable = [(case, checks) for case, checks in planned if checks]
        skipped = tuple(case.id for case, checks in planned if not checks)

        started_at, start = datetime.now(UTC), time.perf_counter()
        results = await asyncio.gather(*(self._run_case(case, checks) for case, checks in runnable))
        return EvalReport(
            pack=self._pack,
            mode=self._target.mode,
            profile=self._profile,
            started_at=started_at,
            duration_ms=round((time.perf_counter() - start) * 1000),
            results=tuple(results),
            skipped=skipped,
        )

    async def _run_case(self, case: EvalCase, checks: Sequence[Check]) -> CaseResult:
        async with self._limit:
            try:
                outcome = await self._target.run(case)
                results = tuple([await check.evaluate(case, outcome) for check in checks])
            except Exception as exc:  # one broken case must not stop the run
                logger.warning("eval case %s failed to run: %s", case.id, exc)
                return CaseResult(
                    case.id, case.tags, False, (), None, f"{type(exc).__name__}: {exc}"
                )
        return CaseResult(case.id, case.tags, all(r.passed for r in results), results, outcome)
