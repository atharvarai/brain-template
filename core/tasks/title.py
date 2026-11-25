"""Conversation titles: the first task that uses structured output.

It runs on the small tier, because naming a conversation needs no deep reasoning.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from core.llm import Completion, LLMProvider, LLMRequest, Message, Role, complete_structured
from core.observability import Tracer
from core.prompts import Prompt, PromptRef, wrap_untrusted

TITLE_PROMPT = "title"
_MAX_MESSAGES = 6  # the opening of a conversation is enough to name it
_MAX_CHARS_PER_MESSAGE = 500


class ConversationTitle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=80)


@dataclass(frozen=True, slots=True)
class TitleResult:
    title: str
    completion: Completion
    prompt: PromptRef
    trace_id: str


class TitleGenerator:
    def __init__(self, provider: LLMProvider, prompt: Prompt, tracer: Tracer) -> None:
        self._provider = provider
        self._prompt = prompt
        self._tracer = tracer
        self._system = prompt.render()  # no variables: fails at start-up if that changes

    async def generate(self, messages: Sequence[Message]) -> TitleResult:
        transcript = "\n".join(
            f"{message.role.value}: {message.content[:_MAX_CHARS_PER_MESSAGE]}"
            for message in messages[:_MAX_MESSAGES]
        )
        request = LLMRequest(
            system=self._system,
            messages=(Message(Role.USER, wrap_untrusted("conversation", transcript)),),
            max_tokens=200,
        )
        async with self._tracer.trace("title") as trace:
            trace.record_prompts([str(self._prompt.ref)])
            with trace.span("generate"):
                result, completion = await complete_structured(
                    self._provider, request, ConversationTitle
                )
            trace.record_completion(completion)
            title = result.title.strip()
            trace.capture(input=transcript, output=title)
            return TitleResult(title, completion, self._prompt.ref, trace.trace_id)
