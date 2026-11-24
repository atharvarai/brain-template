"""Domain pack: the one folder that holds everything specific to a company."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from core.config.schema import BrainConfig

CONFIG_FILENAME = "brain.yaml"


class PackError(Exception):
    """The domain pack is missing or invalid. The message says what to fix."""


@dataclass(frozen=True)
class DomainPack:
    """A loaded, validated domain pack."""

    root: Path
    config: BrainConfig

    @property
    def system_prompt_path(self) -> Path:
        return self.root / self.config.system_prompt


def load_pack(root: Path) -> DomainPack:
    """Load and validate the pack at ``root``. Raises ``PackError`` with a clear reason."""
    root = root.resolve()
    if not root.is_dir():
        raise PackError(f"domain pack folder not found: {root}")

    config_path = root / CONFIG_FILENAME
    if not config_path.is_file():
        raise PackError(f"{CONFIG_FILENAME} not found in domain pack: {root}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PackError(f"{config_path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise PackError(f"{config_path} must contain a mapping at the top level")

    try:
        config = BrainConfig.model_validate(raw)
    except ValidationError as exc:
        raise PackError(f"{config_path} is invalid:\n{exc}") from exc

    pack = DomainPack(root=root, config=config)
    if not pack.system_prompt_path.is_file():
        raise PackError(f"system prompt file not found: {pack.system_prompt_path}")
    return pack
