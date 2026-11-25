"""Liveness endpoint. Also shows which pack and model profile the process is running."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from core import __version__
from core.api.context import AppContext, get_context

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str  # "degraded" when a configured tool server could not be reached
    version: str
    brain: str
    model_profile: str
    tools: list[str]  # discovered tool names
    tool_servers_down: list[str]


@router.get("/healthz")
def healthz(context: Annotated[AppContext, Depends(get_context)]) -> HealthResponse:
    down = list(context.tools.unavailable)
    return HealthResponse(
        status="degraded" if down else "ok",
        version=__version__,
        brain=context.pack.config.name,
        model_profile=context.model_profile,
        tools=[spec.name for spec in context.tools.specs],
        tool_servers_down=down,
    )
