"""Checks: each expectation in a case becomes one ``Check`` (Strategy).

Deterministic checks are cheap and exact; the rubric check asks a judge model and is the
only one that costs a model call.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from core.evals.cases import EvalCase, Expectations
from core.evals.judge import Judge
from core.evals.targets import EvalOutcome

_DIGIT_GROUP = re.compile(r"(?<=\d),(?=\d)")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Case-folded, single-spaced, with digit grouping removed: 42,500 matches 42500."""
    return _SPACE.sub(" ", _DIGIT_GROUP.sub("", text.casefold())).strip()


def _found(expected: str, sections: tuple[str, ...]) -> bool:
    """``doc.md`` matches any section of that document; ``doc.md#Heading`` needs that section
    (a nested heading path such as ``Parent > Heading`` matches on its last part)."""
    if "#" not in expected:
        return any(section.split("#", 1)[0] == expected for section in sections)
    source, heading = expected.split("#", 1)
    return any(
        s_source == source and (s_heading == heading or s_heading.endswith(f" > {heading}"))
        for s_source, _, s_heading in (section.partition("#") for section in sections)
    )


class CheckKind(StrEnum):
    RETRIEVAL = "retrieval"  # needs only the retrieved documents
    ANSWER = "answer"  # needs the model's answer


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


class Check(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> CheckKind: ...

    async def evaluate(self, case: EvalCase, outcome: EvalOutcome) -> CheckResult: ...


@dataclass(frozen=True, slots=True)
class ContainsAll:
    phrases: tuple[str, ...]
    name: str = "contains_all"
    kind: CheckKind = CheckKind.ANSWER

    async def evaluate(self, case: EvalCase, outcome: EvalOutcome) -> CheckResult:
        text = normalize(outcome.answer)
        missing = [p for p in self.phrases if normalize(p) not in text]
        return CheckResult(self.name, not missing, f"missing {missing}" if missing else "")


@dataclass(frozen=True, slots=True)
class ContainsAny:
    phrases: tuple[str, ...]
    name: str = "contains_any"
    kind: CheckKind = CheckKind.ANSWER

    async def evaluate(self, case: EvalCase, outcome: EvalOutcome) -> CheckResult:
        text = normalize(outcome.answer)
        found = any(normalize(p) in text for p in self.phrases)
        return CheckResult(self.name, found, "" if found else f"none of {list(self.phrases)}")


@dataclass(frozen=True, slots=True)
class Excludes:
    phrases: tuple[str, ...]
    name: str = "excludes"
    kind: CheckKind = CheckKind.ANSWER

    async def evaluate(self, case: EvalCase, outcome: EvalOutcome) -> CheckResult:
        text = normalize(outcome.answer)
        present = [p for p in self.phrases if normalize(p) in text]
        return CheckResult(self.name, not present, f"contains {present}" if present else "")


@dataclass(frozen=True, slots=True)
class Retrieves:
    sources: tuple[str, ...]
    name: str = "retrieves"
    kind: CheckKind = CheckKind.RETRIEVAL

    async def evaluate(self, case: EvalCase, outcome: EvalOutcome) -> CheckResult:
        missing = [s for s in self.sources if not _found(s, outcome.retrieved)]
        return CheckResult(self.name, not missing, f"not retrieved {missing}" if missing else "")


@dataclass(frozen=True, slots=True)
class Cites:
    sources: tuple[str, ...]
    name: str = "cites"
    kind: CheckKind = CheckKind.ANSWER

    async def evaluate(self, case: EvalCase, outcome: EvalOutcome) -> CheckResult:
        missing = [s for s in self.sources if not _found(s, outcome.cited)]
        return CheckResult(self.name, not missing, f"not cited {missing}" if missing else "")


@dataclass(frozen=True, slots=True)
class Rubric:
    rubric: str
    judge: Judge | None  # None is fine when only retrieval checks will run
    name: str = "rubric"
    kind: CheckKind = CheckKind.ANSWER

    async def evaluate(self, case: EvalCase, outcome: EvalOutcome) -> CheckResult:
        if self.judge is None:
            raise ValueError("this case has a rubric but no judge model was configured")
        verdict = await self.judge.grade(case.question, outcome.answer, self.rubric)
        return CheckResult(self.name, verdict.passed, verdict.reason)


@dataclass(frozen=True, slots=True)
class Calls:
    tools: tuple[str, ...]  # empty: the answer must not call any tool
    name: str = "calls"
    kind: CheckKind = CheckKind.ANSWER

    async def evaluate(self, case: EvalCase, outcome: EvalOutcome) -> CheckResult:
        called = outcome.tools_called
        if not self.tools:
            return CheckResult(self.name, not called, f"called {list(called)}" if called else "")
        missing = [tool for tool in self.tools if tool not in called]
        detail = f"did not call {missing} (called {list(called)})" if missing else ""
        return CheckResult(self.name, not missing, detail)


def build_checks(expect: Expectations, judge: Judge | None) -> list[Check]:
    """Factory: one check per expectation present."""
    checks: list[Check] = []
    if expect.retrieves:
        checks.append(Retrieves(expect.retrieves))
    if expect.contains_all:
        checks.append(ContainsAll(expect.contains_all))
    if expect.contains_any:
        checks.append(ContainsAny(expect.contains_any))
    if expect.excludes:
        checks.append(Excludes(expect.excludes))
    if expect.cites:
        checks.append(Cites(expect.cites))
    if expect.calls is not None:
        checks.append(Calls(expect.calls))
    if expect.rubric:
        checks.append(Rubric(expect.rubric, judge))
    return checks


def applicable(checks: Sequence[Check], kinds: frozenset[CheckKind]) -> list[Check]:
    return [check for check in checks if check.kind in kinds]
