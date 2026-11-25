"""Routers pick the model tier for one question.

A small model answers a plain lookup ("what does the basic plan cost?") as well as a large one,
at a fraction of the price; reasoning about the user's situation, follow-ups and tool use
need the stronger tier. The evals decide whether a router is good enough: pass rate must
hold while cost per question drops.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from core.config import ModelTier
from core.llm import (
    Completion,
    LLMError,
    LLMProvider,
    LLMRequest,
    Message,
    Role,
    complete_structured,
)
from core.prompts import Prompt, wrap_untrusted

ROUTER_PROMPT = "router"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    tier: ModelTier
    reason: str
    completion: Completion | None = None  # the router's own model call, for cost tracking


class Router(Protocol):
    async def route(self, messages: Sequence[Message]) -> RouteDecision: ...


class FixedRouter:
    def __init__(self, tier: ModelTier) -> None:
        self._tier = tier

    async def route(self, messages: Sequence[Message]) -> RouteDecision:
        return RouteDecision(self._tier, "fixed")


_PERSONAL = re.compile(r"\b(my|me|mine|i|i'm|i've|i'd|myself)\b", re.IGNORECASE)
_REASONING = re.compile(
    r"\b(if|why|should|would|could|compare|difference|versus|vs|better|eligible|calculate"
    r"|after|before|still|explain|both|whether|unless|instead|cheaper|worth)\b",
    re.IGNORECASE,
)
_LOOKUP_START = re.compile(
    r"^\s*(what|when|where|who|which|how (much|long|many|often)|is|are|does|do|can)\b",
    re.IGNORECASE,
)


class HeuristicRouter:
    """Free and instant. Sends a question to the small tier only when it is a short,
    single, impersonal lookup; everything else goes to the default tier."""

    def __init__(self, default: ModelTier, max_words: int = 15) -> None:
        self._default = default
        self._max_words = max_words

    async def route(self, messages: Sequence[Message]) -> RouteDecision:
        text = messages[-1].content
        if len(messages) > 1:
            return RouteDecision(self._default, "follow-up: needs the conversation")
        if _PERSONAL.search(text):
            return RouteDecision(self._default, "about the user: may need their records")
        if _REASONING.search(text) or text.count("?") > 1:
            return RouteDecision(self._default, "needs reasoning")
        if len(text.split()) > self._max_words or not _LOOKUP_START.search(text):
            return RouteDecision(self._default, "not a short lookup")
        return RouteDecision(ModelTier.SMALL, "short lookup")


class _Classification(BaseModel):
    reason: str = Field(description="One short sentence.")
    tier: Literal["small", "mid", "large"]


class ModelRouter:
    """The small model reads the question and picks a tier. One cheap extra call per
    question; falls back to the default tier if the call fails."""

    def __init__(self, provider: LLMProvider, prompt: Prompt, default: ModelTier) -> None:
        self._provider = provider
        self._prompt = prompt
        self._default = default

    async def route(self, messages: Sequence[Message]) -> RouteDecision:
        if len(messages) > 1:
            return RouteDecision(self._default, "follow-up: needs the conversation")
        request = LLMRequest(
            system=self._prompt.render(),
            messages=(Message(Role.USER, wrap_untrusted("question", messages[-1].content)),),
            max_tokens=150,
        )
        try:
            result, completion = await complete_structured(self._provider, request, _Classification)
        except LLMError as exc:
            return RouteDecision(self._default, f"router unavailable: {type(exc).__name__}")
        return RouteDecision(ModelTier(result.tier), result.reason, completion)
