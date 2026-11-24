"""Application factory: the composition root where the layers are wired together."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core import __version__
from core.api import chat, health
from core.api.context import AppContext
from core.config import PackError, Settings, load_pack
from core.llm import (
    ModelGateway,
    ProviderFactory,
    ProviderRequestError,
    ProviderUnavailableError,
    default_factory,
)
from core.pipeline import Pipeline

logger = logging.getLogger(__name__)


def build_context(settings: Settings, provider_factory: ProviderFactory) -> AppContext:
    """Load the pack and build every layer. Fails fast on bad configuration."""
    pack = load_pack(settings.pack)
    models = pack.config.models
    profile = settings.model_profile or models.active
    if profile not in models.profiles:
        raise PackError(
            f"model profile '{profile}' is not defined in {pack.root}; "
            f"known profiles: {sorted(models.profiles)}"
        )
    gateway = ModelGateway.from_config(models, profile, provider_factory)
    system_prompt = pack.system_prompt_path.read_text(encoding="utf-8").strip()
    return AppContext(pack=pack, model_profile=profile, pipeline=Pipeline(gateway, system_prompt))


async def _provider_unavailable(_: Request, exc: Exception) -> JSONResponse:
    logger.error("no provider could answer: %s", exc)
    return JSONResponse(
        status_code=503, content={"detail": "The assistant is temporarily unavailable."}
    )


async def _provider_rejected(_: Request, exc: Exception) -> JSONResponse:
    logger.error("provider rejected the request: %s", exc)
    return JSONResponse(
        status_code=502, content={"detail": "The model provider rejected the request."}
    )


def create_app(
    settings: Settings | None = None, provider_factory: ProviderFactory | None = None
) -> FastAPI:
    """Build the FastAPI app.

    Production reads ``Settings`` from the environment and uses the default provider factory.
    Tests pass their own settings and a factory that builds fake providers.
    """
    context = build_context(settings or Settings(), provider_factory or default_factory())
    app = FastAPI(title=f"{context.pack.config.name} Brain", version=__version__)
    app.state.context = context
    app.include_router(health.router)
    app.include_router(chat.router)
    app.add_exception_handler(ProviderUnavailableError, _provider_unavailable)
    app.add_exception_handler(ProviderRequestError, _provider_rejected)
    return app
