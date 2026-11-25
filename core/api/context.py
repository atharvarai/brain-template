"""Application context: the objects built once at start-up and injected into routes."""

from dataclasses import dataclass
from typing import cast

from fastapi import Request

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


def get_context(request: Request) -> AppContext:
    """FastAPI dependency: routes receive the context instead of reaching for globals."""
    return cast(AppContext, request.app.state.context)
