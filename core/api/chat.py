"""Chat endpoint. Streams Server-Sent Events by default, or returns one JSON body."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any, Self

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from core.access import Caller
from core.agent import ToolActivity
from core.api.context import AppContext, get_caller, get_context
from core.api.schemas import (
    CallMetadata,
    ChatMessage,
    GuardFlagOut,
    PendingActionOut,
    SourceOut,
    ToolActivityOut,
)
from core.llm import LLMError, Role, TextDelta
from core.pipeline import Answer, AnswerStatus, PipelineEvent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = True

    @model_validator(mode="after")
    def _ends_with_user(self) -> Self:
        if self.messages[-1].role is not Role.USER:
            raise ValueError("the last message must have role 'user'")
        return self


class ChatMetadata(CallMetadata):
    """Everything about an answer except its text: sent as the final stream event."""

    sources: list[SourceOut]
    tools: list[ToolActivityOut]
    status: AnswerStatus
    # Set when status is awaiting_approval: decide each pending action at
    # POST /v1/runs/{run_id}/decisions to carry on.
    run_id: str | None
    pending: list[PendingActionOut]
    guard_flags: list[GuardFlagOut]
    # True when a guardrail replaced the model's answer. A streaming client must then show
    # the text of the `replace` event instead of what it streamed.
    replaced: bool

    @classmethod
    def from_answer(cls, answer: Answer) -> Self:
        base = CallMetadata.build(answer.completion, answer.prompts, answer.trace_id)
        return cls(
            sources=[SourceOut.from_source(source) for source in answer.sources],
            tools=[ToolActivityOut.from_activity(activity) for activity in answer.tools],
            status=answer.status,
            run_id=answer.run_id,
            pending=[PendingActionOut.from_call(call) for call in answer.pending],
            guard_flags=[
                GuardFlagOut(guard=f.guard, detail=f.detail, blocked=f.blocked)
                for f in answer.flags
            ],
            replaced=answer.replaced,
            **base.model_dump(),
        )


class ChatResponse(ChatMetadata):
    text: str


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def event_stream(events: AsyncIterator[PipelineEvent]) -> AsyncIterator[str]:
    """delta events carry text, tool events report each tool run, a replace event carries
    text a guardrail put in place of the answer, an approval event lists actions waiting
    for a decision, one done event carries metadata, and an error event ends the stream."""
    try:
        async for event in events:
            if isinstance(event, TextDelta):
                yield _sse("delta", {"text": event.text})
            elif isinstance(event, ToolActivity):
                yield _sse("tool", ToolActivityOut.from_activity(event).model_dump())
            else:
                metadata = ChatMetadata.from_answer(event)
                if event.replaced or event.status is AnswerStatus.BLOCKED:
                    # Whatever streamed before is void: show this text instead.
                    yield _sse("replace", {"text": event.completion.text})
                if metadata.status is AnswerStatus.AWAITING_APPROVAL:
                    yield _sse("approval", metadata.model_dump(include={"run_id", "pending"}))
                yield _sse("done", metadata.model_dump(mode="json"))
    except LLMError:
        # Headers are already sent, so the failure has to travel inside the stream.
        logger.exception("chat stream failed")
        yield _sse("error", {"message": "The assistant is temporarily unavailable."})


def sse_response(events: AsyncIterator[PipelineEvent]) -> StreamingResponse:
    return StreamingResponse(
        event_stream(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    context: Annotated[AppContext, Depends(get_context)],
    caller: Annotated[Caller, Depends(get_caller)],
) -> ChatResponse | StreamingResponse:
    pipeline = context.pipeline
    messages = [message.to_message() for message in body.messages]
    if body.stream:
        return sse_response(pipeline.stream(messages, caller=caller))
    answer = await pipeline.answer(messages, caller=caller)
    metadata = ChatMetadata.from_answer(answer)
    return ChatResponse(text=answer.completion.text, **metadata.model_dump())
