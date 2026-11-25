"""Application factory: the composition root where the layers are wired together."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core import __version__
from core.api import chat, feedback, health, title
from core.api.context import AppContext
from core.config import ModelTier, PackError, Settings, load_pack
from core.llm import (
    ModelGateway,
    ProviderFactory,
    ProviderRequestError,
    ProviderUnavailableError,
    StructuredOutputError,
    default_factory,
)
from core.observability import (
    FanOutSink,
    LoggingTraceSink,
    PriceBook,
    SqliteTraceStore,
    Tracer,
)
from core.pipeline import Pipeline
from core.prompts import ChainedPromptStore, FilePromptStore, build_system_prompt, builtin_store
from core.rag import NullRetriever, Retriever, build_rag
from core.tasks.title import TITLE_PROMPT, TitleGenerator
from core.tools import build_tools

RAG_TURN_PROMPT = "rag_turn"

logger = logging.getLogger(__name__)


TRACES_FILE = "traces.sqlite3"


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

    # The pack's prompts win over the built-in skeletons, so a company can override any of them.
    prompts = ChainedPromptStore(
        [FilePromptStore(pack.prompts_dir, pack.config.prompts.pinned), builtin_store()]
    )
    store = SqliteTraceStore(pack.state_dir(settings.data_dir) / TRACES_FILE)
    tracer = Tracer(
        FanOutSink([store, LoggingTraceSink()]),
        PriceBook(models.prices),
        capture_content=pack.config.observability.capture_content,
        attributes={"pack": pack.config.name, "profile": profile},
    )
    rag = build_rag(pack, settings.data_dir)
    retriever: Retriever = NullRetriever()
    if rag is not None:
        retriever = rag.retriever
        if rag.store.count() == 0:
            logger.warning("the document index is empty: run `make ingest` to build it")
    tools = build_tools(pack.config.tools, pack.root)

    return AppContext(
        pack=pack,
        model_profile=profile,
        gateway=gateway,
        prompts=prompts,
        tracer=tracer,
        traces=store,
        feedback=store,
        pipeline=Pipeline(
            gateway,
            build_system_prompt(prompts, pack.config),
            retriever,
            turn_prompt=prompts.get(RAG_TURN_PROMPT),
            tracer=tracer,
            top_k=rag.top_k if rag else 1,
            tools=tools,
            max_tool_calls=pack.config.tools.max_calls_per_turn,
        ),
        title_generator=TitleGenerator(
            gateway.for_tier(ModelTier.SMALL), prompts.get(TITLE_PROMPT), tracer
        ),
        tools=tools,
    )


def _configure_logging() -> None:
    """Make the core's own log lines visible unless the host already set logging up."""
    if not logging.getLogger().handlers:
        logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("core").setLevel(logging.INFO)


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


async def _invalid_model_output(_: Request, exc: Exception) -> JSONResponse:
    logger.error("model output did not match the schema: %s", exc)
    return JSONResponse(
        status_code=502, content={"detail": "The model did not return a usable result."}
    )


def create_app(
    settings: Settings | None = None, provider_factory: ProviderFactory | None = None
) -> FastAPI:
    """Build the FastAPI app.

    Production reads ``Settings`` from the environment and uses the default provider factory.
    Tests pass their own settings and a factory that builds fake providers.
    """
    _configure_logging()
    context = build_context(settings or Settings(), provider_factory or default_factory())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Tool servers are connected for the life of the process and discovered once.
        async with context.tools:
            yield

    app = FastAPI(title=f"{context.pack.config.name} Brain", version=__version__, lifespan=lifespan)
    app.state.context = context
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(title.router)
    app.include_router(feedback.router)
    app.add_exception_handler(ProviderUnavailableError, _provider_unavailable)
    app.add_exception_handler(ProviderRequestError, _provider_rejected)
    app.add_exception_handler(StructuredOutputError, _invalid_model_output)
    return app
