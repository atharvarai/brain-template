"""Application factory: the composition root where the layers are wired together."""

from fastapi import FastAPI

from core import __version__
from core.api import health
from core.api.context import AppContext
from core.config import PackError, Settings, load_pack


def build_context(settings: Settings) -> AppContext:
    """Load the pack and resolve start-up choices. Fails fast on bad configuration."""
    pack = load_pack(settings.pack)
    profile = settings.model_profile or pack.config.models.active
    if profile not in pack.config.models.profiles:
        raise PackError(
            f"model profile '{profile}' is not defined in {pack.root}; "
            f"known profiles: {sorted(pack.config.models.profiles)}"
        )
    return AppContext(pack=pack, model_profile=profile)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app. Tests pass their own ``Settings``; production reads the env."""
    context = build_context(settings or Settings())
    app = FastAPI(title=f"{context.pack.config.name} Brain", version=__version__)
    app.state.context = context
    app.include_router(health.router)
    return app
