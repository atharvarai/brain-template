"""Feedback endpoint: a thumbs up or down on one answer, joined to its trace."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from core.access import Caller
from core.api.context import AppContext, get_caller, get_context
from core.observability import Feedback, Rating

router = APIRouter(prefix="/v1", tags=["feedback"])


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=1, max_length=64)
    rating: Rating
    comment: str = Field(default="", max_length=2000)


@router.post("/feedback", status_code=204)
async def feedback(
    body: FeedbackRequest,
    context: Annotated[AppContext, Depends(get_context)],
    caller: Annotated[Caller, Depends(get_caller)],
) -> Response:
    if context.access.authenticator.required:
        # Only the user who got the answer may rate it; anyone else sees no such answer.
        trace = context.traces.get(body.trace_id)
        if trace is None or trace.attributes.get("user") != caller.user.id:
            raise HTTPException(status_code=404, detail="No answer with that trace_id.")
    stored = context.feedback.add_feedback(Feedback(body.trace_id, body.rating, body.comment))
    if not stored:
        raise HTTPException(status_code=404, detail="No answer with that trace_id.")
    return Response(status_code=204)
