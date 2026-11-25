"""Eval case files and the checks built from them."""

import json
from pathlib import Path

import pytest

from core.evals import (
    CheckKind,
    EvalCase,
    EvalError,
    EvalOutcome,
    Expectations,
    Verdict,
    build_checks,
    load_cases,
    normalize,
)
from core.llm import Role
from tests.conftest import EDTECH_PACK

pytestmark = pytest.mark.anyio


def write(path: Path, *lines: object) -> Path:
    path.write_text(
        "\n".join(x if isinstance(x, str) else json.dumps(x) for x in lines), encoding="utf-8"
    )
    return path


def case() -> EvalCase:
    """Checks get the case only for its question; its own expectations do not matter here."""
    return EvalCase.model_validate({"id": "c1", "question": "Q?", "expect": {"excludes": ["-"]}})


def outcome(answer: str = "", retrieved: tuple[str, ...] = (), cited: tuple[str, ...] = ()):
    return EvalOutcome(answer=answer, retrieved=retrieved, cited=cited, latency_ms=1)


def test_cases_load_with_history_tags_and_comments(tmp_path: Path) -> None:
    path = write(
        tmp_path / "cases.jsonl",
        "// comments and blank lines are skipped",
        "",
        {
            "id": "follow-up",
            "question": "and how much is it?",
            "history": [
                {"role": "user", "content": "tell me about anvils"},
                {"role": "assistant", "content": "We sell anvils."},
            ],
            "tags": ["pricing"],
            "expect": {"contains_any": ["120"]},
        },
    )
    [loaded] = load_cases(path)
    assert loaded.tags == ("pricing",)
    assert [m.role for m in loaded.messages()] == [Role.USER, Role.ASSISTANT, Role.USER]
    assert loaded.messages()[-1].content == "and how much is it?"


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("{not json", "line 1"),
        ({"id": "a", "question": "q", "expect": {}}, "at least one expectation"),
        ({"id": "a b", "question": "q", "expect": {"excludes": ["x"]}}, "line 1"),
        ({"id": "a", "question": "q", "expect": {"typo": ["x"]}}, "typo"),
        (
            {
                "id": "a",
                "question": "q",
                "history": [{"role": "user", "content": "hi"}],
                "expect": {"excludes": ["x"]},
            },
            "assistant turn",
        ),
    ],
)
def test_bad_cases_are_reported_with_the_line(tmp_path: Path, line: object, message: str) -> None:
    with pytest.raises(EvalError, match=message):
        load_cases(write(tmp_path / "cases.jsonl", line))


def test_duplicate_ids_and_empty_or_missing_files_are_errors(tmp_path: Path) -> None:
    good = {"id": "a", "question": "q", "expect": {"excludes": ["x"]}}
    with pytest.raises(EvalError, match="duplicate case id 'a'"):
        load_cases(write(tmp_path / "dup.jsonl", good, good))
    with pytest.raises(EvalError, match="no cases"):
        load_cases(write(tmp_path / "empty.jsonl", ""))
    with pytest.raises(EvalError, match="no eval cases"):
        load_cases(tmp_path / "missing.jsonl")


def test_the_shipped_example_pack_cases_are_valid() -> None:
    cases = load_cases(EDTECH_PACK / "evals" / "cases.jsonl")
    assert len(cases) >= 20
    root = EDTECH_PACK / "docs"
    docs = {p.relative_to(root).as_posix() for p in root.rglob("*.md")}
    headings = {
        f"{name}#{line.lstrip('#').strip()}"
        for name in docs
        for line in (root / name).read_text(encoding="utf-8").splitlines()
        if line.startswith("## ")
    }
    for loaded in cases:  # a typo in a document or section name could never pass
        for expected in (*loaded.expect.retrieves, *loaded.expect.cites):
            assert expected in (headings if "#" in expected else docs), (loaded.id, expected)


def test_normalize_ignores_case_spacing_and_digit_grouping() -> None:
    assert normalize("  Rs 42,500\n per   Month ") == "rs 42500 per month"
    assert normalize("1,2,3 and a, b") == "123 and a, b"


async def test_text_checks() -> None:
    answer = outcome("The refund is Rs 42,500, paid in 5-7 business days.")
    expectations = {
        "contains_all": ["42500", "business days"],
        "contains_any": ["nope", "REFUND"],
        "excludes": ["guaranteed"],
    }
    results = [
        await c.evaluate(case(), answer) for c in build_checks(Expectations(**expectations), None)
    ]
    assert [(r.name, r.passed) for r in results] == [
        ("contains_all", True),
        ("contains_any", True),
        ("excludes", True),
    ]

    failing = Expectations(contains_all=["42500", "EMI"], contains_any=["x"], excludes=["refund"])
    results = [await c.evaluate(case(), answer) for c in build_checks(failing, None)]
    assert [r.passed for r in results] == [False, False, False]
    assert [r.detail for r in results] == [
        "missing ['EMI']",
        "none of ['x']",
        "contains ['refund']",
    ]


async def test_retrieval_and_citation_checks() -> None:
    expect = Expectations(retrieves=["a.md", "b.md"], cites=["a.md"])
    checks = build_checks(expect, None)
    assert [c.kind for c in checks] == [CheckKind.RETRIEVAL, CheckKind.ANSWER]

    good = outcome(retrieved=("b.md#", "a.md#Fees", "c.md#"), cited=("a.md#Fees",))
    assert all([(await c.evaluate(case(), good)).passed for c in checks])

    bad = outcome(retrieved=("a.md#Intro",), cited=())
    details = [(await c.evaluate(case(), bad)).detail for c in checks]
    assert details == ["not retrieved ['b.md']", "not cited ['a.md']"]


async def test_a_check_can_require_one_section_of_a_document() -> None:
    [check] = build_checks(Expectations(retrieves=["fees.md#Refunds"]), None)
    hit = outcome(retrieved=("fees.md#Prices", "fees.md#Policies > Refunds"))
    miss = outcome(retrieved=("fees.md#Prices", "other.md#Refunds"))
    assert (await check.evaluate(case(), hit)).passed
    assert not (await check.evaluate(case(), miss)).passed


class FakeJudge:
    def __init__(self, passed: bool) -> None:
        self.passed = passed
        self.seen: list[tuple[str, str, str]] = []

    async def grade(self, question: str, answer: str, rubric: str) -> Verdict:
        self.seen.append((question, answer, rubric))
        return Verdict(self.passed, "because")


async def test_the_rubric_check_asks_the_judge() -> None:
    judge = FakeJudge(passed=True)
    [check] = build_checks(Expectations(rubric="Mentions the fee."), judge)
    result = await check.evaluate(case(), outcome("It costs 120."))
    assert (result.name, result.passed, result.detail) == ("rubric", True, "because")
    assert judge.seen == [("Q?", "It costs 120.", "Mentions the fee.")]

    [unjudged] = build_checks(Expectations(rubric="x"), None)
    with pytest.raises(ValueError, match="no judge"):
        await unjudged.evaluate(case(), outcome("x"))
