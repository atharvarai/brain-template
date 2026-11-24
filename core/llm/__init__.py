"""Layer 01, the model: one port, one adapter per vendor, failover and tier routing."""

from core.llm.errors import (
    AllProvidersFailedError,
    LLMConfigError,
    LLMError,
    ProviderRequestError,
    ProviderUnavailableError,
    StructuredOutputError,
)
from core.llm.factory import ProviderFactory, default_factory
from core.llm.fallback import CircuitBreaker, FallbackChain
from core.llm.gateway import ModelGateway
from core.llm.provider import LLMProvider
from core.llm.structured import complete_structured, strict_json_schema
from core.llm.types import (
    Completion,
    LLMRequest,
    Message,
    Role,
    StopReason,
    StreamEvent,
    TextDelta,
    Usage,
)

__all__ = [
    "AllProvidersFailedError",
    "CircuitBreaker",
    "Completion",
    "FallbackChain",
    "LLMConfigError",
    "LLMError",
    "LLMProvider",
    "LLMRequest",
    "Message",
    "ModelGateway",
    "ProviderFactory",
    "ProviderRequestError",
    "ProviderUnavailableError",
    "Role",
    "StopReason",
    "StreamEvent",
    "StructuredOutputError",
    "TextDelta",
    "Usage",
    "complete_structured",
    "default_factory",
    "strict_json_schema",
]
