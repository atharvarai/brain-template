"""LLM-as-judge: grades an answer against a rubric written in plain language."""

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from core.llm import LLMProvider, LLMRequest, Message, Role, complete_structured
from core.observability import Tracer
from core.prompts import Prompt, wrap_untrusted

JUDGE_PROMPT = "judge"
JUDGE_OPERATION = "judge"


class _JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1)  # first, so the verdict follows from the reasoning
    passed: bool


@dataclass(frozen=True, slots=True)
class Verdict:
    passed: bool
    reason: str


class Judge(Protocol):
    async def grade(self, question: str, answer: str, rubric: str) -> Verdict: ...


class LLMJudge:
    def __init__(self, provider: LLMProvider, prompt: Prompt, tracer: Tracer) -> None:
        self._provider = provider
        self._prompt = prompt
        self._system = prompt.render()
        self._tracer = tracer

    async def grade(self, question: str, answer: str, rubric: str) -> Verdict:
        material = "\n".join(
            [
                wrap_untrusted("question", question),
                wrap_untrusted("answer", answer),
                wrap_untrusted("rubric", rubric),
            ]
        )
        request = LLMRequest(
            system=self._system, messages=(Message(Role.USER, material),), max_tokens=400
        )
        async with self._tracer.trace(JUDGE_OPERATION) as trace:
            trace.record_prompts([str(self._prompt.ref)])
            output, completion = await complete_structured(self._provider, request, _JudgeOutput)
            trace.record_completion(completion)
        return Verdict(output.passed, output.reason.strip())
