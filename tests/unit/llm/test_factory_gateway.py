import pytest

from core.config import ModelsConfig, ModelTier, Provider
from core.llm import LLMConfigError, LLMRequest, Message, ModelGateway, ProviderFactory, Role
from core.llm.adapters.claude_code import ClaudeCodeProvider
from core.llm.adapters.fake import FakeProvider
from core.llm.adapters.ollama import OllamaProvider
from core.llm.factory import default_factory
from tests.unit.llm.helpers import request, spec

pytestmark = pytest.mark.anyio


def test_default_factory_builds_an_adapter_for_every_provider(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    factory = default_factory()

    for provider in Provider:
        if provider is Provider.NONE:  # a marker for "no model was called", never built
            continue
        adapter = factory.create(spec(provider, "some-model"))
        assert adapter.name == f"{provider.value}:some-model"

    assert isinstance(factory.create(spec(Provider.FAKE)), FakeProvider)
    assert isinstance(factory.create(spec(Provider.OLLAMA)), OllamaProvider)
    assert isinstance(factory.create(spec(Provider.CLAUDE_CODE)), ClaudeCodeProvider)


def test_unregistered_provider_is_a_config_error() -> None:
    with pytest.raises(LLMConfigError, match="no adapter registered for provider 'openai'"):
        ProviderFactory().create(spec(Provider.OPENAI))


def test_a_new_vendor_is_added_by_registering_a_builder() -> None:
    factory = ProviderFactory()
    factory.register(Provider.OPENAI, lambda s: FakeProvider(s, reply="custom"))

    assert isinstance(factory.create(spec(Provider.OPENAI)), FakeProvider)


async def test_gateway_serves_each_tier_from_the_chosen_profile() -> None:
    def tiers(prefix: str) -> dict[str, list[dict[str, str]]]:
        return {t.value: [{"provider": "fake", "model": f"{prefix}-{t.value}"}] for t in ModelTier}

    models = ModelsConfig.model_validate(
        {"active": "one", "profiles": {"one": tiers("one"), "two": tiers("two")}}
    )
    gateway = ModelGateway.from_config(models, "two", default_factory())

    completion = await gateway.for_tier(ModelTier.SMALL).complete(request())

    assert completion.spec.model == "two-small"


def test_gateway_requires_every_tier() -> None:
    with pytest.raises(ValueError, match="no provider for tier"):
        ModelGateway({ModelTier.SMALL: FakeProvider(spec())})


def test_request_validation() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        LLMRequest(system="s", messages=())
    with pytest.raises(ValueError, match="last message must come from the user"):
        LLMRequest(system="s", messages=(Message(Role.ASSISTANT, "hi"),))


def test_the_none_marker_cannot_be_configured_as_a_model() -> None:
    import pytest
    from pydantic import ValidationError

    from core.config import ModelsConfig

    tiers = {t: [{"provider": "none", "model": "x"}] for t in ("small", "mid", "large")}
    with pytest.raises(ValidationError, match="none"):
        ModelsConfig.model_validate({"active": "p", "profiles": {"p": tiers}})
