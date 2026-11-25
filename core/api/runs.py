"""Decisions on actions waiting for approval. The run carries on from where it paused."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.access import Caller
from core.agent import Decision, DecisionError, RunClosedError, RunNotFoundError
from core.api.chat import ChatMetadata, ChatResponse, sse_response
from core.api.context import AppContext, get_caller, get_context
from core.pipeline import PipelineEvent

router = APIRouter(prefix="/v1", tags=["runs"])


class DecisionIn(BaseModel):
    call_id: str
    approve: bool
    note: str = Field(default="", max_length=500)  # passed to the model, e.g. why it was declined


class DecisionsRequest(BaseModel):
    decisions: list[DecisionIn] = Field(min_length=1)
    stream: bool = True


@router.post(
    "/runs/{run_id}/decisions",
    response_model=ChatResponse,
    responses={404: {"description": "No such paused run, or it expired"},
               409: {"description": "The run was already resumed"},
               422: {"description": "Decisions do not match the pending actions"}},
)  # fmt: skip
async def decide(
    run_id: str,
    body: DecisionsRequest,
    context: Annotated[AppContext, Depends(get_context)],
    caller: Annotated[Caller, Depends(get_caller)],
) -> ChatResponse | StreamingResponse:
    """Approve or decline every pending action of a paused run, then get the answer.

    Each run can be decided once. Approved actions run exactly as they were shown.
    """
    decisions = [Decision(d.call_id, d.approve, d.note) for d in body.decisions]
    pipeline = context.pipeline
    try:
        if body.stream:
            events = pipeline.resume_stream(run_id, decisions, caller)
            first = await anext(events)  # surfaces 404/409/422 before the stream starts
            return sse_response(_prepend(first, events))
        answer = await pipeline.resume(run_id, decisions, caller)
    except RunNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RunClosedError as exc:
        raise HTTPException(409, str(exc)) from exc
    except DecisionError as exc:
        raise HTTPException(422, str(exc)) from exc
    metadata = ChatMetadata.from_answer(answer)
    return ChatResponse(text=answer.completion.text, **metadata.model_dump())


async def _prepend(
    first: PipelineEvent, rest: AsyncIterator[PipelineEvent]
) -> AsyncIterator[PipelineEvent]:
    yield first
    async for event in rest:
        yield event
