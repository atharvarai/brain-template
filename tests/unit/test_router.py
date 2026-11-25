"""Routers pick a tier; the pipeline answers on it and a paused run keeps it."""

import json
from pathlib import Path

import pytest

from core.config import ModelSpec, ModelTier, Provider
from core.llm import Message, ModelGateway, ProviderUnavailableError, Role
from core.llm.adapters.fake import FakeProvider
from core.prompts import builtin_store
from core.router import FixedRouter, HeuristicRouter, ModelRouter
from tests.unit.agent.helpers import ask, build, calls

pytestmark = pytest.mark.anyio

ROUTER = HeuristicRouter(ModelTier.MID)


@pytest.mark.parametrize(
    "question",
    [
        "What does the basic plan cost?",
        "How long is the return window?",
        "When are the live classes?",
        "Is there a mobile app?",
    ],
)
async def test_short_impersonal_lookups_go_to_the_small_tier(question: str) -> None:
    assert (await ROUTER.route([Message(Role.USER, question)])).tier is ModelTier.SMALL


@pytest.mark.parametrize(
    "question",
    [
        "Can I still get a refund if I attended two classes?",  # about the user, reasoning
        "What is my progress in the course?",  # personal: may need their records
        "Why is the advanced plan more expensive than the basic one?",  # reasoning
        "Compare the two plans",  # not a lookup
        "What does it cost? And how long is it?",  # two questions
        "Tell me everything about every plan you sell, with prices and durations please",
    ],
)
async def test_anything_else_stays_on_the_default_tier(question: str) -> None:
    assert (await ROUTER.route([Message(Role.USER, question)])).tier is ModelTier.MID


async def test_follow_ups_stay_on_the_default_tier() -> None:
    history = [Message(Role.USER, "Tell me about plan A"), Message(Role.ASSISTANT, "...")]
    decision = await ROUTER.route([*history, Message(Role.USER, "What does it cost?")])
    assert decision.tier is ModelTier.MID


def small_model(reply: str | None = None, error: Exception | None = None) -> ModelRouter:
    provider = FakeProvider(
        ModelSpec(provider=Provider.FAKE, model="small"), reply=reply, error=error
    )
    return ModelRouter(provider, builtin_store().get("router"), ModelTier.MID)


async def test_the_model_router_uses_the_small_models_choice_and_costs_it() -> None:
    router = small_model(json.dumps({"reason": "a single price", "tier": "small"}))

    decision = await router.route([Message(Role.USER, "What does the basic plan cost?")])

    assert decision.tier is ModelTier.SMALL
    assert decision.completion is not None and decision.completion.usage.input_tokens > 0


async def test_the_model_router_falls_back_to_the_default_when_it_cannot_answer() -> None:
    router = small_model(error=ProviderUnavailableError("down"))

    decision = await router.route([Message(Role.USER, "What does the basic plan cost?")])

    assert decision.tier is ModelTier.MID
    assert "unavailable" in decision.reason


def tiered_gateway() -> tuple[ModelGateway, dict[ModelTier, FakeProvider]]:
    providers = {
        tier: FakeProvider(
            ModelSpec(provider=Provider.FAKE, model=tier.value),
            replies=[calls("cancel_order"), f"answered by {tier.value}"],
        )
        for tier in ModelTier
    }
    return ModelGateway(providers), providers


async def test_the_pipeline_answers_on_the_chosen_tier_and_a_resume_keeps_it(
    pack_dir: Path,
) -> None:
    from core.agent import Decision

    gateway, providers = tiered_gateway()
    pipeline, _, _, store = build(
        pack_dir, [], router=FixedRouter(ModelTier.LARGE), gateway=gateway
    )

    paused = await pipeline.answer(ask("cancel my order"))
    assert paused.run_id is not None and paused.tier is ModelTier.LARGE
    done = await pipeline.resume(paused.run_id, [Decision(paused.pending[0].id, approve=True)])

    assert done.completion.text == "answered by large"
    assert done.tier is ModelTier.LARGE
    assert len(providers[ModelTier.LARGE].requests) == 2
    assert providers[ModelTier.MID].requests == []
    assert store.recent(1)[0].attributes["tier"] == "large"
