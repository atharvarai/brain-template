"""Schema of a domain pack's ``brain.yaml``.

Kept deliberately small: a setting is added here only when a milestone needs it.
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.config.enums import (
    EmbedderKind,
    GroundingCheck,
    GuardAction,
    InjectionDetector,
    JwtAlgorithm,
    ModelTier,
    PiiKind,
    Provider,
    RerankerKind,
    RouterKind,
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
    """List price of a model in US dollars per million tokens. Used to cost each trace.

    Cache rates are optional: without them, cached tokens are costed at the input rate, so
    a missing rate can only overstate the cost, never hide it.
    """

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float | None = Field(default=None, ge=0)
    cache_write: float | None = Field(default=None, ge=0)

    def cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        fresh = max(input_tokens - cache_read_tokens - cache_write_tokens, 0)
        read = self.input if self.cache_read is None else self.cache_read
        write = self.input if self.cache_write is None else self.cache_write
        total = (
            fresh * self.input
            + cache_read_tokens * read
            + cache_write_tokens * write
            + output_tokens * self.output
        )
        return total / 1_000_000


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
            if any(spec.provider is Provider.NONE for chain in profile.values() for spec in chain):
                raise ValueError(f"profile '{name}': provider 'none' is not a real provider")
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
    # Names of the brain's own environment variables the server may see (for example the
    # secret it checks user tokens with). Nothing else is passed on.
    pass_env: tuple[str, ...] = ()


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


class AgentConfig(_Strict):
    """Budgets for answering one question with tools, and which tool calls need a person's
    approval first.

    A step is one model call. When a budget runs out the model is asked once more, with no
    tools, to answer from what it has. Tools whose server marks them read-only run at once;
    any other tool pauses the run for approval unless it is listed in ``auto_approve``.
    """

    max_steps: int = Field(default=6, ge=1, le=20)
    max_tokens: int = Field(default=50_000, ge=1000)  # input + output, summed over steps
    max_tool_calls: int = Field(default=8, ge=1, le=50)
    auto_approve: tuple[str, ...] = ()  # tool names that may change data without asking
    approval_ttl_minutes: int = Field(default=60, ge=1)  # a paused run expires after this


class JwtConfig(_Strict):
    """How the brain checks the user token the company's backend sends with each request."""

    algorithm: JwtAlgorithm = JwtAlgorithm.HS256
    secret_env: str | None = None  # HS256: name of the environment variable with the secret
    public_key_file: str | None = None  # RS256: PEM file, relative to the pack root
    issuer: str | None = None  # checked when set
    audience: str | None = None  # checked when set
    roles_claim: str = "roles"  # claim holding the user's roles, a list of strings
    leeway_seconds: int = Field(default=30, ge=0)

    @model_validator(mode="after")
    def _has_key(self) -> Self:
        if self.algorithm is JwtAlgorithm.HS256 and not self.secret_env:
            raise ValueError("HS256 needs secret_env")
        if self.algorithm is JwtAlgorithm.RS256 and not self.public_key_file:
            raise ValueError("RS256 needs public_key_file")
        return self


class AccessConfig(_Strict):
    """Who may read which documents and call which tools. Leave the section out for a brain
    with no access control: everyone is anonymous, sees unlabelled documents only, and may
    call every tool."""

    jwt: JwtConfig
    policy: str = "policy.yaml"  # roles to readable labels and callable tools, in the pack
    # Demo users that `core.cli token` and the evals may sign tokens for (HS256 only).
    dev_users: str | None = None


class PiiConfig(_Strict):
    """Personal data found in text is replaced by a placeholder like ``[EMAIL]`` before it
    is stored in traces, logs or the audit log. The model still sees what the user wrote."""

    kinds: tuple[PiiKind, ...] = (PiiKind.EMAIL, PiiKind.PHONE, PiiKind.CARD, PiiKind.IP_ADDRESS)


class InjectionConfig(_Strict):
    """Prompt-injection checks: on the user's message, and on retrieved documents and tool
    results (text the brain did not write but the model will read)."""

    detector: InjectionDetector = InjectionDetector.HEURISTIC
    action: GuardAction = GuardAction.BLOCK  # for the user's message
    scan_sources: bool = True  # suspicious documents are dropped, tool results marked
    blocked_message: str = (
        "I can't help with that request. I can answer questions about our services, "
        "policies and your account."
    )


class GroundingConfig(_Strict):
    """Checks that the answer sticks to what it was given."""

    check: GroundingCheck = GroundingCheck.NUMBERS
    action: GuardAction = GuardAction.FLAG


class PromptLeakConfig(_Strict):
    """Stops answers that repeat the system prompt's own wording."""

    action: GuardAction = GuardAction.BLOCK
    min_words: int = Field(default=8, ge=4)  # a run of this many words copied = a leak
    blocked_message: str = (
        "I can't share my instructions, but I'm happy to help with your question."
    )


class GuardrailsConfig(_Strict):
    pii: PiiConfig = Field(default_factory=PiiConfig)
    injection: InjectionConfig = Field(default_factory=InjectionConfig)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    prompt_leak: PromptLeakConfig = Field(default_factory=PromptLeakConfig)


class SemanticCacheConfig(_Strict):
    """Reuse the answer to an earlier question that means the same. Two questions match only
    if their embeddings are this close and they contain exactly the same figures, so
    "refund on day 5" never gets the answer for "refund on day 25"."""

    # Off by default: small embedders can score near misses (same words, opposite meaning)
    # above true paraphrases. Turn it on only after `core.cli cache-check` shows a threshold
    # with no false hits on the pack's own question pairs.
    enabled: bool = False
    threshold: float = Field(default=0.95, gt=0, le=1)  # cosine similarity


class CacheConfig(_Strict):
    """Answer cache. Entries are shared only between users who may read the same documents,
    and expire when the prompts, the documents or the model profile change. Answers that
    used tools or tripped a guardrail are never cached: they belong to one user."""

    exact: bool = True
    semantic: SemanticCacheConfig = Field(default_factory=SemanticCacheConfig)
    ttl_hours: float = Field(default=24, gt=0)


class RouterConfig(_Strict):
    """Which model tier answers each question. Easy lookups on the small tier are much
    cheaper; the evals show whether quality holds."""

    kind: RouterKind = RouterKind.FIXED
    default_tier: ModelTier = ModelTier.MID  # for fixed, and whenever a router is unsure


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
    # With access control on: the demo user (from access.dev_users) cases run as by default.
    default_user: str | None = None


class BrainConfig(_Strict):
    """Everything a company declares about its brain."""

    name: str = Field(min_length=1)
    description: str = ""
    models: ModelsConfig
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    rag: RagConfig | None = None
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    access: AccessConfig | None = None
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    cache: CacheConfig | None = None  # off unless configured
    router: RouterConfig = Field(default_factory=RouterConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    evals: EvalsConfig = Field(default_factory=EvalsConfig)
