"""Reranking as a second pass, and stopwords in keyword search."""

import numpy as np
import pytest

from core.config import RerankerConfig, RerankerKind
from core.rag import (
    Chunk,
    FastEmbedReranker,
    OverlapReranker,
    RerankingRetriever,
    ScoredChunk,
    SqliteChunkStore,
    build_reranker,
)
from core.rag.stopwords import ENGLISH

pytestmark = pytest.mark.anyio


def chunk(source: str, text: str, heading: str = "") -> Chunk:
    return Chunk(source=source, position=0, title=source, heading=heading, text=text)


class FixedRetriever:
    """First pass that returns a fixed list and remembers how many it was asked for."""

    def __init__(self, hits: list[ScoredChunk]) -> None:
        self.hits = hits
        self.asked: list[int] = []

    async def search(
        self, query: str, k: int, allowed_labels: frozenset[str] | None
    ) -> list[ScoredChunk]:
        self.asked.append(k)
        return self.hits[:k]


async def test_the_reranker_reorders_the_first_pass_and_keeps_k() -> None:
    first = FixedRetriever(
        [
            ScoredChunk(chunk("a.md", "anvils are heavy"), 0.9),
            ScoredChunk(chunk("b.md", "how to return a damaged anvil for free"), 0.8),
            ScoredChunk(chunk("c.md", "return shipping"), 0.7),
        ]
    )
    retriever = RerankingRetriever(first, OverlapReranker(), candidates=10)

    hits = await retriever.search("return damaged anvil", 2, None)

    assert first.asked == [10]  # the second pass sees more candidates than it returns
    assert [hit.chunk.source for hit in hits] == ["b.md", "c.md"]
    assert hits[0].score == 1.0  # scores now come from the reranker


async def test_nothing_to_rerank_returns_nothing() -> None:
    retriever = RerankingRetriever(FixedRetriever([]), OverlapReranker())
    assert await retriever.search("anything", 3, None) == []


def test_overlap_reranker_scores_shared_words() -> None:
    assert OverlapReranker().score("Return policy", ["the return window", "x"]) == [0.5, 0.0]
    assert OverlapReranker().score("!!", ["a"]) == [0.0]


def test_build_reranker_follows_the_config() -> None:
    assert isinstance(build_reranker(RerankerConfig(kind=RerankerKind.OVERLAP)), OverlapReranker)
    built = build_reranker(RerankerConfig(model="some/model"))
    assert isinstance(built, FastEmbedReranker)
    assert built.name == "fastembed:some/model"  # loaded lazily: nothing downloaded yet


def store_with(*texts: str, stopwords: frozenset[str] = frozenset()) -> SqliteChunkStore:
    store = SqliteChunkStore(":memory:", stopwords)
    for index, text in enumerate(texts):
        source = f"d{index}.md"
        store.upsert_document(source, "h", [chunk(source, text)], np.ones((1, 2), np.float32))
    return store


def test_stopwords_do_not_decide_keyword_ranking() -> None:
    texts = (
        "what do you want to do and what can I do for you to do it",  # filler only
        "eligibility conditions for placement",
    )
    query = "what do I need to do for placement"
    plain = store_with(*texts).keyword_search(query, 5, None)
    filtered = store_with(*texts, stopwords=ENGLISH).keyword_search(query, 5, None)

    assert plain[0].chunk.source == "d0.md"  # filler words win without a stopword list
    assert [hit.chunk.source for hit in filtered] == ["d1.md"]


def test_a_query_of_only_stopwords_still_searches() -> None:
    store = store_with("to be or not to be", stopwords=ENGLISH)
    assert len(store.keyword_search("to be", 5, None)) == 1


def test_the_corpus_version_moves_on_every_change(tmp_path) -> None:
    import numpy as np

    from core.rag import Chunk
    from core.rag.store import SqliteChunkStore

    store = SqliteChunkStore(tmp_path / "rag.sqlite3")
    store.bind_embedder("hash")
    start = store.version()
    chunk = Chunk("a.md", 0, "A", "", "text")
    store.upsert_document("a.md", "h1", [chunk], np.ones((1, 4), dtype=np.float32))
    after_add = store.version()
    store.delete_document("a.md")

    assert start < after_add < store.version()
    assert SqliteChunkStore(tmp_path / "rag.sqlite3").version() == store.version()  # persisted
