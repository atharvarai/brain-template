"""Request and response bodies shared by the API routes."""

from typing import Self

from pydantic import BaseModel, Field

from core.llm import Completion, Message, Role
from core.prompts import PromptRef
from core.rag import Source


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)

    def to_message(self) -> Message:
        return Message(self.role, self.content)


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int


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
    """What produced an answer: recorded per request so any reply can be traced back."""

    provider: str
    model: str
    stop_reason: str
    usage: UsageOut
    latency_ms: int
    prompts: list[PromptRefOut]

    @classmethod
    def build(cls, completion: Completion, prompts: tuple[PromptRef, ...]) -> Self:
        return cls(
            provider=completion.spec.provider.value,
            model=completion.spec.model,
            stop_reason=completion.stop_reason.value,
            usage=UsageOut(
                input_tokens=completion.usage.input_tokens,
                output_tokens=completion.usage.output_tokens,
            ),
            latency_ms=completion.latency_ms,
            prompts=[PromptRefOut.from_ref(ref) for ref in prompts],
        )
