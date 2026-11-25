"""Schema of a domain pack's ``brain.yaml``.

Kept deliberately small: a setting is added here only when a milestone needs it.
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.config.enums import (
    EmbedderKind,
    ModelTier,
    Provider,
    RerankerKind,
    StopwordList,
    ToolTransport,
)


class _Strict(BaseModel):
    """Base for config models: immutable, and unknown keys are an error, not ignored."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelSpec(_Strict):
    """One concrete model at one provider."""

    provider: Provider
    model: str = Field(min_length=1)


# For each tier, an ordered fallback chain: the first spec is tried first.
ModelProfile = dict[ModelTier, tuple[ModelSpec, ...]]


class Price(_Strict):
    """List price of a model in US dollars per million tokens. Used to cost each trace."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input + output_tokens * self.output) / 1_000_000


class ModelsConfig(_Strict):
    """Named model profiles (for example dev / prod), each mapping every tier to a chain."""

    active: str
    profiles: dict[str, ModelProfile]
    # Keyed by model id. A model with no entry is traced with an unknown cost, not zero.
    prices: dict[str, Price] = Field(default_factory=dict)

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


class RerankerConfig(_Strict):
    """Optional second pass: a cross-encoder reads the question with each candidate and
    re-orders them. Slower than the first pass, so it only sees the top ``candidates``."""

    kind: RerankerKind = RerankerKind.FASTEMBED
    model: str = "jinaai/jina-reranker-v1-tiny-en"
    candidates: int = Field(default=20, ge=1, le=100)


class RagConfig(_Strict):
    """Retrieval over the company's documents. Leave it out for a brain with no documents."""

    docs: str = "docs"  # folder of Markdown files, relative to the pack root
    top_k: int = Field(default=5, ge=1, le=20)  # chunks given to the model per question
    chunk_chars: int = Field(default=1200, ge=100)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    stopwords: StopwordList = StopwordList.ENGLISH  # dropped from keyword queries
    reranker: RerankerConfig | None = None  # off unless the pack's evals show it helps


class StdioServerConfig(_Strict):
    """An MCP server the brain starts itself. ``python`` as the command means the brain's own
    interpreter; the working directory is the pack root."""

    transport: Literal[ToolTransport.STDIO]
    command: tuple[str, ...] = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)


class HttpServerConfig(_Strict):
    """An MCP server reached over Streamable HTTP, usually run by the company's backend."""

    transport: Literal[ToolTransport.HTTP]
    url: str = Field(pattern=r"^https?://")


McpServerConfig = Annotated[StdioServerConfig | HttpServerConfig, Field(discriminator="transport")]


class ToolsConfig(_Strict):
    """Tools come only from MCP servers: the core never names a tool. Each server's tools are
    discovered at start-up."""

    servers: dict[str, McpServerConfig] = Field(default_factory=dict)  # keyed by a short name
    timeout_seconds: float = Field(default=15, gt=0)  # per tool call
    max_result_chars: int = Field(default=4000, ge=100)  # longer results are cut, with a note
    max_calls_per_turn: int = Field(default=4, ge=1, le=16)


class ObservabilityConfig(_Strict):
    """What each request's trace keeps. Traces always hold timings, tokens, cost, model,
    prompt versions and document ids; the user's text and the answer only if allowed here."""

    capture_content: bool = False


class EvalsConfig(_Strict):
    """The pack's evaluation set and the bar it has to clear."""

    cases: str = "evals/cases.jsonl"  # relative to the pack root
    min_pass_rate: float = Field(default=0.8, ge=0, le=1)  # `eval` exits non-zero below it
    judge_tier: ModelTier = ModelTier.MID  # model that grades rubric checks
    concurrency: int = Field(default=4, ge=1, le=32)


class BrainConfig(_Strict):
    """Everything a company declares about its brain."""

    name: str = Field(min_length=1)
    description: str = ""
    models: ModelsConfig
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    rag: RagConfig | None = None
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    evals: EvalsConfig = Field(default_factory=EvalsConfig)
