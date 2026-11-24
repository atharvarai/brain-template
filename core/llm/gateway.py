"""The model gateway: callers ask for a tier and get a provider with failover built in."""

from collections.abc import Mapping
from typing import Self

from core.config import ModelsConfig, ModelTier
from core.llm.factory import ProviderFactory
from core.llm.fallback import FallbackChain
from core.llm.provider import LLMProvider


class ModelGateway:
    """Maps each ``ModelTier`` to the provider (usually a fallback chain) that serves it."""

    def __init__(self, by_tier: Mapping[ModelTier, LLMProvider]) -> None:
        missing = [tier.value for tier in ModelTier if tier not in by_tier]
        if missing:
            raise ValueError(f"no provider for tier(s): {missing}")
        self._by_tier = dict(by_tier)

    @classmethod
    def from_config(cls, models: ModelsConfig, profile: str, factory: ProviderFactory) -> Self:
        """Build one fallback chain per tier from the chosen profile in brain.yaml."""
        return cls(
            {
                tier: FallbackChain([factory.create(spec) for spec in models.chain(tier, profile)])
                for tier in ModelTier
            }
        )

    def for_tier(self, tier: ModelTier) -> LLMProvider:
        return self._by_tier[tier]
