"""Error taxonomy for the model layer.

The split that matters is "would another provider help?". Outages, overload, timeouts and
rate limits are ``ProviderUnavailableError`` and trigger failover. A rejected request is our
bug or our misconfiguration: it would fail everywhere, so it is surfaced, not hidden.
"""

RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})


class LLMError(Exception):
    """Base class for model layer errors."""


class LLMConfigError(LLMError):
    """The model configuration cannot be satisfied (unknown provider, missing SDK)."""


class ProviderUnavailableError(LLMError):
    """Temporary or provider-side failure. Trying another provider is reasonable."""


class ProviderRequestError(LLMError):
    """The provider rejected the request itself (bad input, bad credentials)."""


class AllProvidersFailedError(ProviderUnavailableError):
    """Every provider in a fallback chain was unavailable."""

    def __init__(self, errors: tuple[str, ...]) -> None:
        super().__init__("all providers failed: " + " | ".join(errors))
        self.errors = errors


def error_for_status(provider: str, status: int, message: str) -> LLMError:
    """Map an HTTP status from any vendor onto the taxonomy above."""
    detail = f"{provider}: HTTP {status}: {message}"
    if status in RETRYABLE_STATUS_CODES or status >= 500:
        return ProviderUnavailableError(detail)
    return ProviderRequestError(detail)
