"""Liveness endpoint. Also shows which pack and model profile the process is running."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from core import __version__
from core.api.context import AppContext, get_context

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    brain: str
    model_profile: str


@router.get("/healthz")
def healthz(context: Annotated[AppContext, Depends(get_context)]) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        brain=context.pack.config.name,
        model_profile=context.model_profile,
    )
