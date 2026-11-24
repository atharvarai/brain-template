"""Builds the right adapter for a ``ModelSpec``.

Adding a vendor means writing one adapter and registering one builder here (Open/Closed):
no other code changes. Vendor SDKs are imported lazily, so a deployment only needs the
SDKs of the providers it actually configures.
"""

from collections.abc import Callable, Mapping

from core.config import ModelSpec, Provider
from core.llm.errors import LLMConfigError
from core.llm.provider import LLMProvider

ProviderBuilder = Callable[[ModelSpec], LLMProvider]


class ProviderFactory:
    """Registry of one builder per ``Provider`` enum value."""

    def __init__(self, builders: Mapping[Provider, ProviderBuilder] | None = None) -> None:
        self._builders: dict[Provider, ProviderBuilder] = dict(builders or {})

    def register(self, provider: Provider, builder: ProviderBuilder) -> None:
        self._builders[provider] = builder

    def create(self, spec: ModelSpec) -> LLMProvider:
        builder = self._builders.get(spec.provider)
        if builder is None:
            raise LLMConfigError(f"no adapter registered for provider '{spec.provider.value}'")
        return builder(spec)


def _missing_sdk(provider: Provider, package: str, exc: ImportError) -> LLMConfigError:
    return LLMConfigError(
        f"provider '{provider.value}' needs the '{package}' package: "
        f"install it with `uv sync --extra {package}` ({exc})"
    )


def _build_anthropic(spec: ModelSpec) -> LLMProvider:
    try:
        from core.llm.adapters.anthropic_adapter import AnthropicProvider
    except ImportError as exc:
        raise _missing_sdk(Provider.ANTHROPIC, "anthropic", exc) from exc
    return AnthropicProvider(spec)


def _build_openai(spec: ModelSpec) -> LLMProvider:
    try:
        from core.llm.adapters.openai_adapter import OpenAIProvider
    except ImportError as exc:
        raise _missing_sdk(Provider.OPENAI, "openai", exc) from exc
    return OpenAIProvider(spec)


def _build_ollama(spec: ModelSpec) -> LLMProvider:
    from core.llm.adapters.ollama import OllamaProvider

    return OllamaProvider(spec)


def _build_claude_code(spec: ModelSpec) -> LLMProvider:
    from core.llm.adapters.claude_code import ClaudeCodeProvider

    return ClaudeCodeProvider(spec)


def _build_fake(spec: ModelSpec) -> LLMProvider:
    from core.llm.adapters.fake import FakeProvider

    return FakeProvider(spec)


def default_factory() -> ProviderFactory:
    """A factory that knows every provider in the ``Provider`` enum."""
    return ProviderFactory(
        {
            Provider.ANTHROPIC: _build_anthropic,
            Provider.OPENAI: _build_openai,
            Provider.OLLAMA: _build_ollama,
            Provider.CLAUDE_CODE: _build_claude_code,
            Provider.FAKE: _build_fake,
        }
    )
