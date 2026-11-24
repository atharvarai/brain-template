"""Chat endpoint. Streams Server-Sent Events by default, or returns one JSON body."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any, Self

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from core.api.context import AppContext, get_context
from core.llm import Completion, LLMError, Message, Role, TextDelta
from core.pipeline import Pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = True

    @model_validator(mode="after")
    def _ends_with_user(self) -> Self:
        if self.messages[-1].role is not Role.USER:
            raise ValueError("the last message must have role 'user'")
        return self

    def to_messages(self) -> list[Message]:
        return [Message(m.role, m.content) for m in self.messages]


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int


class ChatResponse(BaseModel):
    text: str
    provider: str
    model: str
    stop_reason: str
    usage: UsageOut
    latency_ms: int

    @classmethod
    def from_completion(cls, completion: Completion) -> Self:
        return cls(
            text=completion.text,
            provider=completion.spec.provider.value,
            model=completion.spec.model,
            stop_reason=completion.stop_reason.value,
            usage=UsageOut(
                input_tokens=completion.usage.input_tokens,
                output_tokens=completion.usage.output_tokens,
            ),
            latency_ms=completion.latency_ms,
        )


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _event_stream(pipeline: Pipeline, messages: list[Message]) -> AsyncIterator[str]:
    """delta events carry text, one done event carries metadata, error ends the stream."""
    try:
        async for event in pipeline.stream(messages):
            if isinstance(event, TextDelta):
                yield _sse("delta", {"text": event.text})
            else:
                done = ChatResponse.from_completion(event).model_dump(exclude={"text"})
                yield _sse("done", done)
    except LLMError:
        # Headers are already sent, so the failure has to travel inside the stream.
        logger.exception("chat stream failed")
        yield _sse("error", {"message": "The assistant is temporarily unavailable."})


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest, context: Annotated[AppContext, Depends(get_context)]
) -> ChatResponse | StreamingResponse:
    messages = body.to_messages()
    if body.stream:
        return StreamingResponse(
            _event_stream(context.pipeline, messages),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return ChatResponse.from_completion(await context.pipeline.answer(messages))
