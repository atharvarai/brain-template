"""Conversation title endpoint: a small structured-output task on the small tier."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from core.access import Caller
from core.api.context import AppContext, get_caller, get_context
from core.api.schemas import CallMetadata, ChatMessage

router = APIRouter(prefix="/v1", tags=["title"])


class TitleRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)


class TitleResponse(CallMetadata):
    title: str


@router.post("/title")
async def title(
    body: TitleRequest,
    context: Annotated[AppContext, Depends(get_context)],
    _: Annotated[Caller, Depends(get_caller)],  # signed-in users only, when access is on
) -> TitleResponse:
    result = await context.title_generator.generate([m.to_message() for m in body.messages])
    metadata = CallMetadata.build(result.completion, (result.prompt,), result.trace_id)
    return TitleResponse(title=result.title, **metadata.model_dump())
