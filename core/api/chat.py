"""Chat endpoint. Streams Server-Sent Events by default, or returns one JSON body."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any, Self

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from core.api.context import AppContext, get_context
from core.api.schemas import CallMetadata, ChatMessage
from core.llm import LLMError, Message, Role, TextDelta
from core.pipeline import Pipeline

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


class ChatResponse(CallMetadata):
    text: str


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _log_call(metadata: CallMetadata) -> None:
    # Until the observability layer exists, the prompt versions are recorded in the log.
    logger.info(
        "chat answered provider=%s model=%s prompts=%s tokens_in=%d tokens_out=%d ms=%d",
        metadata.provider,
        metadata.model,
        ",".join(f"{p.name}@v{p.version}#{p.checksum}" for p in metadata.prompts),
        metadata.usage.input_tokens,
        metadata.usage.output_tokens,
        metadata.latency_ms,
    )


async def _event_stream(pipeline: Pipeline, messages: list[Message]) -> AsyncIterator[str]:
    """delta events carry text, one done event carries metadata, error ends the stream."""
    try:
        async for event in pipeline.stream(messages):
            if isinstance(event, TextDelta):
                yield _sse("delta", {"text": event.text})
            else:
                metadata = CallMetadata.build(event, pipeline.prompt_refs)
                _log_call(metadata)
                yield _sse("done", metadata.model_dump())
    except LLMError:
        # Headers are already sent, so the failure has to travel inside the stream.
        logger.exception("chat stream failed")
        yield _sse("error", {"message": "The assistant is temporarily unavailable."})


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest, context: Annotated[AppContext, Depends(get_context)]
) -> ChatResponse | StreamingResponse:
    pipeline = context.pipeline
    messages = [message.to_message() for message in body.messages]
    if body.stream:
        return StreamingResponse(
            _event_stream(pipeline, messages),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    completion = await pipeline.answer(messages)
    metadata = CallMetadata.build(completion, pipeline.prompt_refs)
    _log_call(metadata)
    return ChatResponse(text=completion.text, **metadata.model_dump())
