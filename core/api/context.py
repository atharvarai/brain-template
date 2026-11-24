"""Application context: the objects built once at start-up and injected into routes."""

from dataclasses import dataclass
from typing import cast

from fastapi import Request

from core.config import DomainPack


@dataclass(frozen=True)
class AppContext:
    """Composition root output. Later milestones add the pipeline and its layers here."""

    pack: DomainPack
    model_profile: str


def get_context(request: Request) -> AppContext:
    """FastAPI dependency: routes receive the context instead of reaching for globals."""
    return cast(AppContext, request.app.state.context)
