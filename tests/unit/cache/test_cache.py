"""The answer cache on its own and in the pipeline: what may be reused, and by whom."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from core.access import Caller, RolePolicy, User
from core.access.policy import PolicyFile
from core.cache import AnswerCache, SqliteCacheStore, cache_scope, figures
from core.config import CacheConfig
from core.llm import Message, Role
from core.pipeline import AnswerStatus
from core.rag.embedder import HashingEmbedder
from tests.unit.agent.helpers import ask, build, calls

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)


def cache(tmp_path: Path, clock=lambda: NOW, **config: object) -> AnswerCache:
    store = SqliteCacheStore(tmp_path / "cache.sqlite3", now=clock)
    return AnswerCache(store, CacheConfig.model_validate(config), HashingEmbedder())


def q(text: str) -> list[Message]:
    return [Message(Role.USER, text)]


BODY = {"text": "Thirty days.", "sources": [], "provider": "fake", "model": "m", "tier": "mid"}


async def test_an_exact_repeat_hits_regardless_of_case_and_spacing(tmp_path: Path) -> None:
    answers = cache(tmp_path)
    await answers.store(q("How long is the return window?"), "s1", BODY)

    hit = await answers.lookup(q("  how long is the RETURN window  "), "s1")

    assert hit is not None and (hit.kind, hit.answer.body["text"]) == ("exact", "Thirty days.")


async def test_another_scope_never_sees_the_entry(tmp_path: Path) -> None:
    answers = cache(tmp_path)
    await answers.store(q("How long is the return window?"), "s1", BODY)

    assert await answers.lookup(q("How long is the return window?"), "s2") is None


async def test_a_paraphrase_hits_but_different_figures_never_do(tmp_path: Path) -> None:
    answers = cache(tmp_path, semantic={"enabled": True, "threshold": 0.6})
    await answers.store(q("Can I get a refund on day 5 of the course?"), "s1", BODY)

    paraphrase = await answers.lookup(q("can i get refund on day 5 of course"), "s1")
    other_day = await answers.lookup(q("Can I get a refund on day 25 of the course?"), "s1")

    assert paraphrase is not None and paraphrase.kind in {"exact", "semantic"}
    assert other_day is None
    assert figures("Rs 85,000 on day 5") == {"85000", "5"}


async def test_follow_ups_only_match_the_same_conversation(tmp_path: Path) -> None:
    answers = cache(tmp_path, semantic={"enabled": True, "threshold": 0.1})
    history = [Message(Role.USER, "Tell me about plan A"), Message(Role.ASSISTANT, "Plan A...")]
    await answers.store([*history, Message(Role.USER, "How much is it?")], "s1", BODY)

    other = [Message(Role.USER, "Tell me about plan B"), Message(Role.ASSISTANT, "Plan B...")]
    assert await answers.lookup([*other, Message(Role.USER, "How much is it?")], "s1") is None


async def test_entries_expire(tmp_path: Path) -> None:
    clock = {"now": NOW}
    answers = cache(tmp_path, clock=lambda: clock["now"], ttl_hours=1)
    await answers.store(q("hours?"), "s1", BODY)

    clock["now"] = NOW + timedelta(hours=2)

    assert await answers.lookup(q("hours?"), "s1") is None


def test_the_scope_depends_on_every_part() -> None:
    base = cache_scope(labels=[], prompts=["system@v5"], corpus=3)
    assert base == cache_scope(prompts=["system@v5"], corpus=3, labels=[])
    assert base != cache_scope(labels=["internal"], prompts=["system@v5"], corpus=3)
    assert base != cache_scope(labels=[], prompts=["system@v6"], corpus=3)
    assert base != cache_scope(labels=[], prompts=["system@v5"], corpus=4)


# --- in the pipeline --------------------------------------------------------------------------

POLICY = RolePolicy(
    PolicyFile.model_validate(
        {"roles": {"member": {"tools": ["*"]}, "staff": {"read": ["internal"], "tools": ["*"]}}}
    )
)
MEMBER = Caller(User("m1", frozenset({"member"})))
MEMBER_2 = Caller(User("m2", frozenset({"member"})))
STAFF = Caller(User("s1", frozenset({"staff"})))


def cached_pipeline(pack_dir: Path, tmp_path: Path, replies: list, corpus: list[int]):
    return build(
        pack_dir, replies, policy=POLICY,
        cache=cache(tmp_path, semantic={"enabled": False}),
        cache_context=lambda: {"corpus": corpus[0]},
    )  # fmt: skip


async def test_a_repeat_from_someone_with_the_same_access_skips_the_model(
    pack_dir: Path, tmp_path: Path
) -> None:
    pipeline, provider, _, store = cached_pipeline(pack_dir, tmp_path, ["Thirty days."], [1])

    first = await pipeline.answer(ask("return window?"), caller=MEMBER)
    second = await pipeline.answer(ask("return window?"), caller=MEMBER_2)

    assert len(provider.requests) == 1
    assert (first.cached, second.cached) == (None, "exact")
    assert second.completion.text == "Thirty days."
    assert store.recent(1)[0].events[0].name == "cache_hit"


async def test_users_who_read_other_documents_do_not_share(pack_dir: Path, tmp_path: Path) -> None:
    pipeline, provider, _, _ = cached_pipeline(pack_dir, tmp_path, ["answer"], [1])

    await pipeline.answer(ask("margins?"), caller=MEMBER)
    staff = await pipeline.answer(ask("margins?"), caller=STAFF)

    assert staff.cached is None
    assert len(provider.requests) == 2


async def test_answers_that_used_tools_are_never_cached(pack_dir: Path, tmp_path: Path) -> None:
    pipeline, provider, _, _ = cached_pipeline(
        pack_dir, tmp_path, [calls("find_order"), "Shipped.", calls("find_order"), "Shipped."], [1]
    )

    await pipeline.answer(ask("where is my order?"), caller=MEMBER)
    again = await pipeline.answer(ask("where is my order?"), caller=MEMBER)

    assert again.cached is None
    assert len(provider.requests) == 4


async def test_a_document_change_or_opting_out_skips_the_cache(
    pack_dir: Path, tmp_path: Path
) -> None:
    corpus = [1]
    pipeline, provider, _, _ = cached_pipeline(pack_dir, tmp_path, ["v1 answer"], corpus)
    await pipeline.answer(ask("return window?"), caller=MEMBER)

    bypass = await pipeline.answer(ask("return window?"), caller=MEMBER, use_cache=False)
    corpus[0] = 2  # someone re-ran ingest
    after_change = await pipeline.answer(ask("return window?"), caller=MEMBER)

    assert bypass.cached is None and after_change.cached is None
    assert len(provider.requests) == 3


async def test_blocked_and_cached_answers_are_streamed_too(pack_dir: Path, tmp_path: Path) -> None:
    pipeline, _, _, _ = cached_pipeline(pack_dir, tmp_path, ["Thirty days."], [1])
    await pipeline.answer(ask("return window?"), caller=MEMBER)

    events = [e async for e in pipeline.stream(ask("return window?"), caller=MEMBER)]

    assert events[-1].status is AnswerStatus.COMPLETE  # type: ignore[union-attr]
    assert events[-1].cached == "exact"  # type: ignore[union-attr]
    assert events[0].text == "Thirty days."  # type: ignore[union-attr]


def test_the_threshold_sweep_counts_good_and_wrong_reuses() -> None:
    from core.evals.cache_pairs import Pair, PairScore, sweep

    scores = [
        PairScore(Pair("a", "a'", True), 0.95, True),
        PairScore(Pair("b", "b'", True), 0.85, True),
        PairScore(Pair("c", "not c", False), 0.90, True),
        PairScore(Pair("day 5", "day 25", False), 0.99, False),  # figures differ: never served
    ]

    low, mid, high = sweep(scores, [0.80, 0.88, 0.92])

    assert (low.hits, low.false_hits) == (2, 1)
    assert (mid.hits, mid.false_hits) == (1, 1)
    assert (high.hits, high.false_hits) == (1, 0)
