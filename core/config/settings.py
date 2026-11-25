"""Process-level settings read from environment variables (prefix ``BRAIN_``)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from core.config.enums import RouterKind


class Settings(BaseSettings):
    """Deployment settings. Company settings live in the pack's brain.yaml instead."""

    model_config = SettingsConfigDict(env_prefix="BRAIN_", env_file=".env", extra="ignore")

    pack: Path  # BRAIN_PACK: path to the domain pack folder
    model_profile: str | None = None  # BRAIN_MODEL_PROFILE: overrides models.active
    data_dir: Path = Path(".brain")  # BRAIN_DATA_DIR: where the brain keeps its own data
    # BRAIN_ROUTER: overrides brain.yaml's router kind, to compare routers on the same evals
    router: RouterKind | None = None
