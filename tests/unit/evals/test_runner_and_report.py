"""Running cases, the report, and comparing with the previous run."""

import asyncio
import json
from pathlib import Path

import pytest

from core.evals import (
    EvalCase,
    EvalMode,
    EvalOutcome,
    EvalRunner,
    ReportArchive,
    Verdict,
    compare,
    render,
    report_to_dict,
)

pytestmark = pytest.mark.anyio


def make_case(case_id: str, tags: tuple[str, ...] = (), **expect: object) -> EvalCase:
    return EvalCase.model_validate(
        {"id": case_id, "question": f"question {case_id}", "tags": tags, "expect": expect}
    )


class ScriptedTarget:
    """Answers each case from a script; a missing entry raises like a broken model would."""

    def __init__(self, mode: EvalMode, answers: dict[str, str], delay: float = 0) -> None:
        self.mode = mode
        self._answers = answers
        self._delay = delay
        self.running = 0
        self.peak = 0

    async def run(self, case: EvalCase) -> EvalOutcome:
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(self._delay)
        self.running -= 1
        if case.id not in self._answers:
            raise TimeoutError("model timed out")
        return EvalOutcome(
            answer=self._answers[case.id],
            retrieved=("fees.md",),
            cited=(),
            latency_ms=1000,
            trace_id=f"trace-{case.id}",
            input_tokens=100,
            output_tokens=10,
            cost_usd=0.001,
        )


class AlwaysPass:
    async def grade(self, question: str, answer: str, rubric: str) -> Verdict:
        return Verdict(True, "fine")


CASES = [
    make_case("fee", ("pricing",), contains_any=["120"], retrieves=["fees.md"]),
    make_case("wrong", ("pricing",), contains_any=["999"]),
    make_case("judged", ("tone",), rubric="Polite."),
    make_case("crash", ("tone",), excludes=["x"]),
]


async def run(target: ScriptedTarget, cases: list[EvalCase] = CASES, concurrency: int = 4):
    return await EvalRunner(target, AlwaysPass(), "Acme", "test", concurrency).run(cases)


async def test_answer_mode_runs_every_check_and_isolates_broken_cases() -> None:
    target = ScriptedTarget(
        EvalMode.ANSWER, {"fee": "It is 120.", "wrong": "It is 120.", "judged": "Hi"}
    )
    report = await run(target)

    assert [(r.case_id, r.passed) for r in report.results] == [
        ("fee", True),
        ("wrong", False),
        ("judged", True),
        ("crash", False),
    ]
    assert report.results[3].error == "TimeoutError: model timed out"
    assert report.pass_rate == 0.5
    assert report.by_tag() == {"pricing": (1, 2), "tone": (1, 2)}
    assert report.by_check() == {
        "retrieves": (1, 1),
        "contains_any": (1, 2),
        "rubric": (1, 1),
    }
    assert report.tokens == (300, 30)
    assert report.cost_usd == pytest.approx(0.003)
    assert report.latency_ms(0.5) == 1000


async def test_retrieval_mode_runs_only_retrieval_checks_and_skips_the_rest() -> None:
    target = ScriptedTarget(EvalMode.RETRIEVAL, {"fee": ""})
    report = await run(target)

    assert [r.case_id for r in report.results] == ["fee"]
    assert [c.name for c in report.results[0].checks] == ["retrieves"]
    assert report.skipped == ("wrong", "judged", "crash")
    assert report.pass_rate == 1.0


async def test_concurrency_is_bounded() -> None:
    many = [make_case(f"c{i}", excludes=["x"]) for i in range(8)]
    target = ScriptedTarget(EvalMode.ANSWER, {c.id: "ok" for c in many}, delay=0.01)
    await run(target, many, concurrency=3)
    assert target.peak == 3


async def test_reports_are_archived_and_compared_with_the_previous_run(tmp_path: Path) -> None:
    archive = ReportArchive(tmp_path / "evals")
    assert archive.latest("answer") is None

    first = await run(
        ScriptedTarget(EvalMode.ANSWER, {"fee": "120", "wrong": "999", "judged": "x"})
    )
    path = archive.save(first)
    saved = json.loads(path.read_text())
    assert saved == report_to_dict(first)
    assert saved["cases"][0]["trace_id"] == "trace-fee"

    second = await run(
        ScriptedTarget(EvalMode.ANSWER, {"fee": "no", "wrong": "999", "crash": "ok"})
    )
    comparison = compare(second, archive.latest("answer") or {})
    assert comparison.regressions == ("fee", "judged")
    assert comparison.fixes == ("crash",)
    # Rates compare only the cases both runs had, so a filtered run is not misread.
    assert comparison.shared == 4
    assert (comparison.previous_rate, comparison.current_rate) == (0.75, 0.5)

    text = render(second, threshold=0.8, comparison=comparison)
    assert "FAIL  fee" in text
    assert "contains_any: none of ['120']" in text
    assert "BELOW THRESHOLD" in text
    assert "regressions: fee, judged" in text
    assert "fixed: crash" in text
    assert "on the 4 cases both ran: 75.0% then, 50.0% now (-25.0 points)" in text
    assert "$0.0030 at list price" in text
