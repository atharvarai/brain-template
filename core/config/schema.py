"""Schema of a domain pack's ``brain.yaml``.

Kept deliberately small: a setting is added here only when a milestone needs it.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.config.enums import EmbedderKind, ModelTier, Provider


class _Strict(BaseModel):
    """Base for config models: immutable, and unknown keys are an error, not ignored."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelSpec(_Strict):
    """One concrete model at one provider."""

    provider: Provider
    model: str = Field(min_length=1)


# For each tier, an ordered fallback chain: the first spec is tried first.
ModelProfile = dict[ModelTier, tuple[ModelSpec, ...]]


class ModelsConfig(_Strict):
    """Named model profiles (for example dev / prod), each mapping every tier to a chain."""

    active: str
    profiles: dict[str, ModelProfile]

    @model_validator(mode="after")
    def _profiles_are_complete(self) -> Self:
        if self.active not in self.profiles:
            raise ValueError(
                f"active profile '{self.active}' is not defined; "
                f"known profiles: {sorted(self.profiles)}"
            )
        for name, profile in self.profiles.items():
            missing = [tier.value for tier in ModelTier if not profile.get(tier)]
            if missing:
                raise ValueError(f"profile '{name}' has no model for tier(s): {missing}")
        return self

    def chain(self, tier: ModelTier, profile: str | None = None) -> tuple[ModelSpec, ...]:
        """Return the fallback chain for ``tier`` in ``profile`` (default: the active one)."""
        name = profile or self.active
        if name not in self.profiles:
            raise KeyError(f"unknown model profile '{name}'; known: {sorted(self.profiles)}")
        return self.profiles[name][tier]


class PromptsConfig(_Strict):
    """Prompt files live in the pack's ``prompts/`` folder as ``<name>.v<version>.md``.

    The newest version of each prompt is used unless it is pinned here, which is how a
    bad prompt change gets rolled back without deleting the file.
    """

    pinned: dict[str, int] = Field(default_factory=dict)


class EmbedderConfig(_Strict):
    kind: EmbedderKind = EmbedderKind.FASTEMBED
    model: str = "BAAI/bge-small-en-v1.5"  # used by kinds that load a named model


class RagConfig(_Strict):
    """Retrieval over the company's documents. Leave it out for a brain with no documents."""

    docs: str = "docs"  # folder of Markdown files, relative to the pack root
    top_k: int = Field(default=5, ge=1, le=20)  # chunks given to the model per question
    chunk_chars: int = Field(default=1200, ge=100)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)


class BrainConfig(_Strict):
    """Everything a company declares about its brain."""

    name: str = Field(min_length=1)
    description: str = ""
    models: ModelsConfig
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    rag: RagConfig | None = None
