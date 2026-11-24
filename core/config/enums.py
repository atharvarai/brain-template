"""Enums shared across layers. Code refers to these, never to vendor or model strings."""

from enum import StrEnum


class Provider(StrEnum):
    """Who serves the model. Each value has exactly one adapter in ``core.llm``."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    CLAUDE_CODE = "claude_code"  # local `claude -p` CLI on a subscription, no API key
    FAKE = "fake"  # deterministic, for tests


class ModelTier(StrEnum):
    """How capable a model a task needs. Features ask for a tier, never for a model."""

    SMALL = "small"  # classify, route, extract
    MID = "mid"  # everyday answers
    LARGE = "large"  # multi-step reasoning, agent work
