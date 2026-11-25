"""Application context: the objects built once at start-up and injected into routes."""

from dataclasses import dataclass
from typing import cast

from fastapi import HTTPException, Request

from core.access import (
    ANONYMOUS_CALLER,
    AccessControl,
    AuditAction,
    AuditEntry,
    AuditLog,
    AuthError,
    Caller,
    User,
)
from core.config import DomainPack
from core.llm import ModelGateway
from core.observability import FeedbackSink, Tracer, TraceReader
from core.pipeline import Pipeline
from core.prompts import PromptStore
from core.tasks.title import TitleGenerator
from core.tools import ToolRegistry


@dataclass(frozen=True)
class AppContext:
    """Composition root output. Later milestones add more layers behind the pipeline."""

    pack: DomainPack
    model_profile: str
    gateway: ModelGateway
    prompts: PromptStore
    tracer: Tracer
    traces: TraceReader
    feedback: FeedbackSink
    pipeline: Pipeline
    title_generator: TitleGenerator
    tools: ToolRegistry  # connect with ``async with context.tools`` before serving
    access: AccessControl
    audit: AuditLog


def get_context(request: Request) -> AppContext:
    """FastAPI dependency: routes receive the context instead of reaching for globals."""
    return cast(AppContext, request.app.state.context)


def get_caller(request: Request) -> Caller:
    """FastAPI dependency: the user behind the request, from its bearer token.

    With access control off every request is anonymous. With it on, a missing or bad token
    is a 401, and the refusal is written to the audit log.
    """
    context = get_context(request)
    if not context.access.authenticator.required:
        return ANONYMOUS_CALLER
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    token = token.strip() if scheme.lower() == "bearer" else ""
    try:
        user = context.access.authenticator.verify(token or None)
    except AuthError as exc:
        context.audit.record(
            AuditEntry.of(
                AuditAction.AUTH_FAILED, User("unknown"), request.url.path, "rejected",
                detail={"reason": str(exc)[:200]},
            )
        )  # fmt: skip
        raise HTTPException(
            status_code=401,
            detail="A valid bearer token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return Caller(user, token)
