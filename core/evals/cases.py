"""Eval cases: questions a pack expects its brain to handle, with what a good answer shows.

Cases live in the pack as JSON Lines, one case per line::

    {"id": "returns-window", "question": "How long do I have to return it?",
     "tags": ["returns"],
     "expect": {"contains_any": ["30 days"], "retrieves": ["returns.md"],
                "rubric": "States the 30-day window and nothing contradicting it."}}
"""

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from core.llm import Message, Role


class EvalError(Exception):
    """The cases file is missing or invalid. The message says which line and why."""


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Turn(_Strict):
    role: Role
    content: str = Field(min_length=1)


class Expectations(_Strict):
    """What a passing answer shows. Every field given becomes one check."""

    contains_all: tuple[str, ...] = ()  # every phrase appears in the answer
    contains_any: tuple[str, ...] = ()  # at least one phrase appears
    excludes: tuple[str, ...] = ()  # none of these appear (leaks, wrong facts)
    # Documents ("doc.md") or sections ("doc.md#Heading") that must be among those retrieved
    retrieves: tuple[str, ...] = ()
    cites: tuple[str, ...] = ()  # same form: the answer must cite these
    rubric: str | None = None  # graded by a judge model

    @model_validator(mode="after")
    def _not_empty(self) -> Self:
        if not any(
            (
                self.contains_all,
                self.contains_any,
                self.excludes,
                self.retrieves,
                self.cites,
                self.rubric,
            )
        ):
            raise ValueError("a case needs at least one expectation")
        return self


class EvalCase(_Strict):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    question: str = Field(min_length=1)
    history: tuple[Turn, ...] = ()  # earlier turns, oldest first
    tags: tuple[str, ...] = ()
    expect: Expectations

    @model_validator(mode="after")
    def _history_alternates(self) -> Self:
        if self.history and self.history[-1].role is not Role.ASSISTANT:
            raise ValueError("history must end with an assistant turn")
        return self

    def messages(self) -> list[Message]:
        turns = [Message(turn.role, turn.content) for turn in self.history]
        return [*turns, Message(Role.USER, self.question)]


def load_cases(path: Path) -> list[EvalCase]:
    if not path.is_file():
        raise EvalError(f"no eval cases at {path}")
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            case = EvalCase.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise EvalError(f"{path.name} line {number}: {exc}") from exc
        if case.id in seen:
            raise EvalError(f"{path.name} line {number}: duplicate case id '{case.id}'")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise EvalError(f"{path} has no cases")
    return cases
