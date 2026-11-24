"""Retrievers find the chunks most relevant to a question."""

from collections.abc import Sequence
from typing import Protocol

import anyio

from core.rag.embedder import Embedder
from core.rag.store import ChunkStore
from core.rag.types import Chunk, ScoredChunk


class Retriever(Protocol):
    async def search(
        self, query: str, k: int, allowed_labels: frozenset[str] | None
    ) -> list[ScoredChunk]: ...


class NullRetriever:
    """Used when a brain has no documents configured: retrieval finds nothing."""

    async def search(
        self, query: str, k: int, allowed_labels: frozenset[str] | None
    ) -> list[ScoredChunk]:
        return []


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], k: int, rank_constant: int = 60
) -> list[ScoredChunk]:
    """Merge ranked lists whose scores are not comparable (cosine vs BM25).

    Each list contributes 1 / (rank_constant + rank) per chunk, so a chunk found by both
    searches beats one found by only one. Only ranks are used, never the raw scores.
    """
    fused: dict[tuple[str, int], float] = {}
    chunks: dict[tuple[str, int], Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = (hit.chunk.source, hit.chunk.position)
            fused[key] = fused.get(key, 0.0) + 1.0 / (rank_constant + rank)
            chunks[key] = hit.chunk
    best = sorted(fused, key=lambda key: (-fused[key], key))[:k]
    return [ScoredChunk(chunks[key], fused[key]) for key in best]


class HybridRetriever:
    """Semantic search plus keyword search, merged with reciprocal rank fusion.

    Vectors catch paraphrases ("money back" finds "refund"). Keywords catch exact terms that
    embeddings blur, such as product names, codes and numbers.
    """

    def __init__(self, embedder: Embedder, store: ChunkStore, candidates: int = 20) -> None:
        self._embedder = embedder
        self._store = store
        self._candidates = candidates

    async def search(
        self, query: str, k: int, allowed_labels: frozenset[str] | None
    ) -> list[ScoredChunk]:
        # Embedding and SQLite are blocking, so they run off the event loop.
        return await anyio.to_thread.run_sync(self._search, query, k, allowed_labels)

    def _search(
        self, query: str, k: int, allowed_labels: frozenset[str] | None
    ) -> list[ScoredChunk]:
        if not query.strip() or self._store.count() == 0:
            return []
        by_meaning = self._store.vector_search(
            self._embedder.embed_query(query), self._candidates, allowed_labels
        )
        by_keyword = self._store.keyword_search(query, self._candidates, allowed_labels)
        return reciprocal_rank_fusion([by_meaning, by_keyword], k)
