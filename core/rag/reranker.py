"""Reranking: a slower, more precise second pass over the first-pass candidates.

``RerankingRetriever`` is a Decorator: it wraps any ``Retriever`` and has the same
interface, so the pipeline does not know whether reranking is on.
"""

import re
from collections.abc import Sequence
from typing import Any, Protocol

import anyio

from core.config import RerankerConfig, RerankerKind
from core.rag.errors import RagError
from core.rag.retriever import Retriever
from core.rag.types import ScoredChunk

_WORD = re.compile(r"\w+")


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """One relevance score per text; higher is more relevant."""
        ...


class FastEmbedReranker:
    """Local cross-encoder through fastembed. Downloaded on first use, loaded lazily."""

    def __init__(self, model: str) -> None:
        self._model_name = model
        self._model: Any = None

    @property
    def name(self) -> str:
        return f"fastembed:{self._model_name}"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._load().rerank(query, list(texts))]

    def _load(self) -> Any:
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as exc:
                raise RagError(f"the 'fastembed' package is not installed ({exc})") from exc
            self._model = TextCrossEncoder(self._model_name)
        return self._model


class OverlapReranker:
    """Fraction of query words found in the text. No model: for tests only."""

    name = "overlap"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        words = set(_WORD.findall(query.lower()))
        if not words:
            return [0.0] * len(texts)
        return [len(words & set(_WORD.findall(text.lower()))) / len(words) for text in texts]


class RerankingRetriever:
    def __init__(self, inner: Retriever, reranker: Reranker, candidates: int = 20) -> None:
        self._inner = inner
        self._reranker = reranker
        self._candidates = candidates

    async def search(
        self, query: str, k: int, allowed_labels: frozenset[str] | None
    ) -> list[ScoredChunk]:
        pool = await self._inner.search(query, max(k, self._candidates), allowed_labels)
        if not pool:
            return []
        texts = [hit.chunk.searchable_text for hit in pool]
        scores = await anyio.to_thread.run_sync(self._reranker.score, query, texts)
        # Stable sort: ties keep the first-pass order.
        ranked = sorted(zip(scores, pool, strict=True), key=lambda pair: -pair[0])
        return [ScoredChunk(hit.chunk, score) for score, hit in ranked[:k]]


def build_reranker(config: RerankerConfig) -> Reranker:
    match config.kind:
        case RerankerKind.FASTEMBED:
            return FastEmbedReranker(config.model)
        case RerankerKind.OVERLAP:
            return OverlapReranker()
