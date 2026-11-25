"""Vendor-neutral request and response types. Adapters translate to and from these."""

from collections.abc import Mapping
from dataclasses import dataclass, field
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
    TOOL_USE = "tool_use"  # the model is waiting for the results of the tools it asked for
    OTHER = "other"


class ToolChoice(StrEnum):
    AUTO = "auto"  # the model decides whether to call a tool
    NONE = "none"  # tools stay visible (history may mention them) but must not be called


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A function the model may ask the application to run. Comes from tool discovery."""

    name: str
    description: str
    input_schema: Mapping[str, Any]  # JSON schema of the arguments object


@dataclass(frozen=True, slots=True)
class ToolCall:
    """The model asking for one tool to be run. ``id`` pairs the call with its result."""

    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What came back from running a ``ToolCall``, sent to the model in the next user turn."""

    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Message:
    """One conversation turn. An assistant turn may carry tool calls; the user turn after it
    carries their results. Each adapter maps this onto its vendor's own shape."""

    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()

    def __post_init__(self) -> None:
        if self.tool_calls and self.role is not Role.ASSISTANT:
            raise ValueError("only an assistant message can carry tool calls")
        if self.tool_results and self.role is not Role.USER:
            raise ValueError("only a user message can carry tool results")


@dataclass(frozen=True, slots=True)
class LLMRequest:
    system: str
    messages: tuple[Message, ...]
    max_tokens: int = 2048
    # When set, the answer text must be JSON matching this schema. Adapters pass it to the
    # vendor's native structured-output feature; see ``core.llm.structured``.
    json_schema: dict[str, Any] | None = None
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: ToolChoice = ToolChoice.AUTO

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("an LLM request needs at least one message")
        if self.messages[-1].role is not Role.USER:
            raise ValueError("the last message must come from the user")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.tools and self.json_schema is not None:
            raise ValueError("a request cannot ask for both tools and structured output")
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError(f"tool names must be unique: {sorted(names)}")

    @property
    def offers_tools(self) -> bool:
        """True when the model may call a tool in its reply."""
        return bool(self.tools) and self.tool_choice is ToolChoice.AUTO


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts for one call. ``input_tokens`` is every input token, cached or not;
    the cache fields say how many of them were read from or written to the vendor's prompt
    cache (billed at different rates)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True, slots=True)
class Completion:
    """A finished answer, plus what the observability layer needs to know about it."""

    text: str
    spec: ModelSpec  # which provider and model actually answered (after any fallback)
    usage: Usage
    stop_reason: StopReason
    latency_ms: int
    tool_calls: tuple[ToolCall, ...] = field(default=())  # set when stop_reason is TOOL_USE


@dataclass(frozen=True, slots=True)
class TextDelta:
    """One piece of streamed text."""

    text: str


# A stream yields zero or more TextDelta events and then exactly one Completion.
StreamEvent = TextDelta | Completion
