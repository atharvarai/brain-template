"""Enums shared across layers. Code refers to these, never to vendor or model strings."""

from enum import StrEnum


class Provider(StrEnum):
    """Who serves the model. Each value has exactly one adapter in ``core.llm``."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    CLAUDE_CODE = "claude_code"  # local `claude -p` CLI on a subscription, no API key
    FAKE = "fake"  # deterministic, for tests


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
