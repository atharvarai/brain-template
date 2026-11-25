"""Enums shared across layers. Code refers to these, never to vendor or model strings."""

from enum import StrEnum


class Provider(StrEnum):
    """Who serves the model. Each value has exactly one adapter in ``core.llm``."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    CLAUDE_CODE = "claude_code"  # local `claude -p` CLI on a subscription, no API key
    FAKE = "fake"  # deterministic, for tests
    NONE = "none"  # no model was called (a guardrail answered); never configured in a profile


class EmbedderKind(StrEnum):
    """Which embedding implementation turns text into vectors for retrieval."""

    FASTEMBED = "fastembed"  # local ONNX model, free and offline
    HASHING = "hashing"  # word hashing, no model: tests and smoke runs only


class RerankerKind(StrEnum):
    """Which model re-scores retrieval candidates against the question."""

    FASTEMBED = "fastembed"  # local ONNX cross-encoder, free and offline
    OVERLAP = "overlap"  # word overlap, no model: tests and smoke runs only


class StopwordList(StrEnum):
    """Words ignored by keyword search. Pick the language of the pack's documents."""

    ENGLISH = "english"
    NONE = "none"  # keep every word: for languages without a list yet


class ModelTier(StrEnum):
    """How capable a model a task needs. Features ask for a tier, never for a model."""

    SMALL = "small"  # classify, route, extract
    MID = "mid"  # everyday answers
    LARGE = "large"  # multi-step reasoning, agent work


class ToolTransport(StrEnum):
    """How the brain reaches an MCP server."""

    STDIO = "stdio"  # the brain starts the server as a child process: local development
    HTTP = "http"  # Streamable HTTP to a server the company runs: production


class JwtAlgorithm(StrEnum):
    """How user tokens are signed."""

    HS256 = "HS256"  # shared secret: simple, fine when one backend issues and checks tokens
    RS256 = "RS256"  # key pair: the brain holds only the public key


class PiiKind(StrEnum):
    """Personal data the redactor finds. Regional kinds are opt-in per pack."""

    EMAIL = "email"
    PHONE = "phone"
    CARD = "card"  # payment card numbers that pass the Luhn check
    IP_ADDRESS = "ip_address"
    IN_PAN = "in_pan"  # Indian Permanent Account Number
    IN_AADHAAR = "in_aadhaar"  # Indian Aadhaar number (Verhoeff check digit)


class InjectionDetector(StrEnum):
    HEURISTIC = "heuristic"  # phrase patterns: free and instant, English only
    MODEL = "model"  # a small-tier model classifies the message: slower, any language
    OFF = "off"


class GuardAction(StrEnum):
    FLAG = "flag"  # record it on the trace and the answer, change nothing
    BLOCK = "block"  # refuse (input) or replace the answer (output)


class GroundingCheck(StrEnum):
    NUMBERS = "numbers"  # every figure in the answer must appear in what it was given
    OFF = "off"


class RouterKind(StrEnum):
    """How the brain picks a model tier for each question."""

    FIXED = "fixed"  # always the default tier
    HEURISTIC = "heuristic"  # word rules: free and instant, English only
    MODEL = "model"  # the small tier classifies the question: one extra cheap call
