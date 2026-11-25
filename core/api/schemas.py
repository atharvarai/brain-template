"""Request and response bodies shared by the API routes."""

from typing import Any, Self

from pydantic import BaseModel, Field

from core.agent import ToolActivity
from core.llm import Completion, Message, Role, ToolCall
from core.prompts import PromptRef
from core.rag import Source


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)

    def to_message(self) -> Message:
        return Message(self.role, self.content)


class UsageOut(BaseModel):
    input_tokens: int  # all input, cached or not
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class PromptRefOut(BaseModel):
    name: str
    version: int
    checksum: str

    @classmethod
    def from_ref(cls, ref: PromptRef) -> Self:
        return cls(name=ref.name, version=ref.version, checksum=ref.checksum)


class SourceOut(BaseModel):
    """A document section the answer could draw on. ``index`` matches citations like [1]."""

    index: int
    source: str
    title: str
    section: str
    cited: bool

    @classmethod
    def from_source(cls, source: Source) -> Self:
        return cls(
            index=source.index,
            source=source.source,
            title=source.title,
            section=source.section,
            cited=source.cited,
        )


class CallMetadata(BaseModel):
    """What produced an answer. ``trace_id`` finds the full trace and is what feedback cites."""

    trace_id: str
    provider: str
    model: str
    stop_reason: str
    usage: UsageOut
    latency_ms: int
    prompts: list[PromptRefOut]

    @classmethod
    def build(cls, completion: Completion, prompts: tuple[PromptRef, ...], trace_id: str) -> Self:
        return cls(
            trace_id=trace_id,
            provider=completion.spec.provider.value,
            model=completion.spec.model,
            stop_reason=completion.stop_reason.value,
            usage=UsageOut(
                input_tokens=completion.usage.input_tokens,
                output_tokens=completion.usage.output_tokens,
                cache_read_tokens=completion.usage.cache_read_tokens,
                cache_write_tokens=completion.usage.cache_write_tokens,
            ),
            latency_ms=completion.latency_ms,
            prompts=[PromptRefOut.from_ref(ref) for ref in prompts],
        )


class ToolActivityOut(BaseModel):
    """A tool the brain ran while answering. Arguments and results are not echoed back."""

    name: str
    ok: bool
    latency_ms: int
    step: int

    @classmethod
    def from_activity(cls, activity: ToolActivity) -> Self:
        return cls(
            name=activity.name, ok=activity.ok, latency_ms=activity.latency_ms, step=activity.step
        )


class PendingActionOut(BaseModel):
    """An action waiting for approval, shown exactly as it will run if approved."""

    call_id: str
    tool: str
    arguments: dict[str, Any]

    @classmethod
    def from_call(cls, call: ToolCall) -> Self:
        return cls(call_id=call.id, tool=call.name, arguments=dict(call.arguments))


class GuardFlagOut(BaseModel):
    """Something a guardrail noticed. ``blocked`` means it changed the outcome."""

    guard: str
    detail: str
    blocked: bool
