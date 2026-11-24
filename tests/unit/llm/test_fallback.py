import pytest

from core.config import Provider
from core.llm import (
    AllProvidersFailedError,
    CircuitBreaker,
    FallbackChain,
    ProviderRequestError,
    ProviderUnavailableError,
)
from core.llm.adapters.fake import FakeProvider
from tests.unit.llm.helpers import BrokenStreamProvider, collect, request, spec

pytestmark = pytest.mark.anyio


def down(model: str = "down") -> FakeProvider:
    return FakeProvider(spec(model=model), error=ProviderUnavailableError(f"{model} is down"))


def up(model: str = "up", reply: str = "answer") -> FakeProvider:
    return FakeProvider(spec(Provider.OLLAMA, model), reply=reply)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_uses_the_first_provider_when_it_works() -> None:
    first, second = up("a", "from a"), up("b", "from b")

    completion = await FallbackChain([first, second]).complete(request())

    assert completion.text == "from a"
    assert second.requests == []


async def test_fails_over_when_a_provider_is_unavailable() -> None:
    completion = await FallbackChain([down(), up(reply="from backup")]).complete(request())

    assert completion.text == "from backup"
    assert completion.spec.provider is Provider.OLLAMA  # callers can see who answered


async def test_does_not_fail_over_on_a_rejected_request() -> None:
    rejected = FakeProvider(spec(), error=ProviderRequestError("bad request"))
    backup = up()

    with pytest.raises(ProviderRequestError):
        await FallbackChain([rejected, backup]).complete(request())
    assert backup.requests == []


async def test_reports_every_failure_when_all_providers_are_down() -> None:
    with pytest.raises(AllProvidersFailedError) as caught:
        await FallbackChain([down("a"), down("b")]).complete(request())

    assert caught.value.errors == ("a is down", "b is down")


async def test_circuit_opens_then_allows_a_trial_call_after_the_cooldown() -> None:
    clock = FakeClock()
    flaky = down("flaky")
    chain = FallbackChain(
        [flaky, up()],
        breaker_factory=lambda: CircuitBreaker(2, cooldown_seconds=30, clock=clock),
    )

    for _ in range(2):  # two failures open the circuit
        await chain.complete(request())
    assert len(flaky.requests) == 2

    await chain.complete(request())  # open: the flaky provider is skipped
    assert len(flaky.requests) == 2

    clock.now = 31.0  # half-open: one trial call goes through
    await chain.complete(request())
    assert len(flaky.requests) == 3


def test_circuit_closes_again_after_a_success() -> None:
    breaker = CircuitBreaker(failure_threshold=2)
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()

    assert breaker.allows_request()  # failures were not consecutive


async def test_stream_fails_over_before_the_first_delta() -> None:
    broken = BrokenStreamProvider("broken", 0, ProviderUnavailableError("broken is down"))

    text, final = await collect(FallbackChain([broken, up(reply="from backup")]).stream(request()))

    assert text.strip() == "from backup"
    assert final.text == "from backup"


async def test_stream_does_not_fail_over_after_text_was_sent() -> None:
    broken = BrokenStreamProvider("broken", 2, ProviderUnavailableError("dropped mid-answer"))
    backup = up()

    with pytest.raises(ProviderUnavailableError, match="dropped mid-answer"):
        await collect(FallbackChain([broken, backup]).stream(request()))
    assert backup.requests == []


def test_an_empty_chain_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one provider"):
        FallbackChain([])
