"""Vendor-neutral request and response types. Adapters translate to and from these."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from core.config import ModelSpec


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class StopReason(StrEnum):
    """Why generation stopped, normalised across vendors."""

    END = "end"  # finished normally
    MAX_TOKENS = "max_tokens"  # hit the output cap: the text is truncated
    REFUSAL = "refusal"  # the model or a safety filter declined
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class LLMRequest:
    system: str
    messages: tuple[Message, ...]
    max_tokens: int = 2048
    # When set, the answer text must be JSON matching this schema. Adapters pass it to the
    # vendor's native structured-output feature; see ``core.llm.structured``.
    json_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("an LLM request needs at least one message")
        if self.messages[-1].role is not Role.USER:
            raise ValueError("the last message must come from the user")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class Completion:
    """A finished answer, plus what the observability layer needs to know about it."""

    text: str
    spec: ModelSpec  # which provider and model actually answered (after any fallback)
    usage: Usage
    stop_reason: StopReason
    latency_ms: int


@dataclass(frozen=True, slots=True)
class TextDelta:
    """One piece of streamed text."""

    text: str


# A stream yields zero or more TextDelta events and then exactly one Completion.
StreamEvent = TextDelta | Completion
