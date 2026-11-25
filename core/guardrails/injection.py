"""Prompt-injection detection: text that tries to give the model new orders.

Two detectors behind one interface. The heuristic one matches known phrasings: free and
instant, English only, easy to rephrase around. The model one asks a small model and
catches other languages and paraphrases, at the cost of a model call per message. Neither
is a wall: the system prompt's rules and wrapping untrusted text are the other layers.
"""

import re
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

from core.llm import Completion, LLMProvider, LLMRequest, Message, Role, complete_structured
from core.prompts import Prompt, wrap_untrusted

INJECTION_PROMPT = "injection"


@dataclass(frozen=True, slots=True)
class InjectionVerdict:
    suspicious: bool
    reasons: tuple[str, ...] = ()
    completion: Completion | None = None  # set when a model was asked, for cost tracking


class InjectionDetector(Protocol):
    async def inspect(self, text: str) -> InjectionVerdict: ...


_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in {
        "override_instructions": (
            r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}"
            r"\b(previous|prior|above|earlier|all|any|your|the|system)\b[^.\n]{0,30}"
            r"\b(instructions?|rules|prompts?|guidelines|directions|restrictions)\b"
        ),
        "reveal_prompt": (
            r"\b(reveal|print|show|repeat|output|display|leak|dump|tell me)\b[^.\n]{0,40}"
            r"\b(system prompt|(your|hidden|initial|original) (instructions|rules|prompt))"
        ),
        "role_override": (
            r"\byou are now\b|\bfrom now on,? you (are|will)\b|\bdeveloper mode\b|\bjailbreak"
            r"|\bact as (an? )?(unrestricted|unfiltered|uncensored)|\bDAN\b"
        ),
        "fake_turn_marker": (
            r"(^|\n)\s*(system|assistant|developer)\s*:|<\s*/?\s*(system|instructions?)\s*>"
            r"|\[/?INST\]|<\|im_start\|>"
        ),
        "new_instructions": r"\b(new|updated|real|actual) (system )?(instructions|rules)\s*:",
    }.items()
}


class HeuristicInjectionDetector:
    async def inspect(self, text: str) -> InjectionVerdict:
        return self.inspect_now(text)

    def inspect_now(self, text: str) -> InjectionVerdict:
        """Synchronous form, for scanning documents and tool results."""
        reasons = tuple(name for name, pattern in _PATTERNS.items() if pattern.search(text))
        return InjectionVerdict(bool(reasons), reasons)


class _Classification(BaseModel):
    reason: str = Field(description="One short sentence.")
    suspicious: bool


class ModelInjectionDetector:
    def __init__(self, provider: LLMProvider, prompt: Prompt) -> None:
        self._provider = provider
        self._prompt = prompt

    async def inspect(self, text: str) -> InjectionVerdict:
        request = LLMRequest(
            system=self._prompt.render(),
            messages=(Message(Role.USER, wrap_untrusted("message", text)),),
            max_tokens=200,
        )
        result, completion = await complete_structured(self._provider, request, _Classification)
        reasons = ("model: " + result.reason,) if result.suspicious else ()
        return InjectionVerdict(result.suspicious, reasons, completion)


class NoInjectionDetector:
    async def inspect(self, text: str) -> InjectionVerdict:
        return InjectionVerdict(False)
