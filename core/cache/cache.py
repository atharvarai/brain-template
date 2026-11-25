"""The answer cache: an exact match first, then (optionally) a question that means the same.

What may be shared is decided by the *scope*: a hash of everything an answer depends on
besides the question. Users who can read the same documents, under the same prompts,
model profile and document versions, share entries; anyone else gets their own.
"""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import anyio

from core.cache.store import CachedAnswer, CacheStore
from core.config import CacheConfig
from core.llm import Message
from core.rag import Embedder

_SPACE = re.compile(r"\s+")
_FIGURE = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass(frozen=True, slots=True)
class CacheHit:
    answer: CachedAnswer
    kind: str  # "exact" or "semantic"
    similarity: float = 1.0


def cache_scope(**parts: Any) -> str:
    """One opaque string for everything an answer depends on besides the question."""
    canonical = json.dumps(parts, sort_keys=True, default=sorted)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def normalize(text: str) -> str:
    return _SPACE.sub(" ", text.casefold()).strip().rstrip("?.! ")


def figures(text: str) -> frozenset[str]:
    """Every number in the text. Similar questions with different figures differ."""
    return frozenset(f.replace(",", "") for f in _FIGURE.findall(text))


class AnswerCache:
    def __init__(
        self, store: CacheStore, config: CacheConfig, embedder: Embedder | None = None
    ) -> None:
        self._store = store
        self._config = config
        semantic = config.semantic.enabled and embedder is not None
        self._embedder = embedder if semantic else None

    async def lookup(self, messages: Sequence[Message], scope: str) -> CacheHit | None:
        key = self._key(messages, scope)
        if self._config.exact and (hit := self._store.get(key)) is not None:
            return CacheHit(hit, "exact")
        if self._embedder is None or len(messages) != 1:  # follow-ups depend on history
            return None
        question = messages[-1].content
        vector = await anyio.to_thread.run_sync(self._embedder.embed_query, normalize(question))
        found = self._store.nearest(scope, vector, figures(question))
        if found is None or found[1] < self._config.semantic.threshold:
            return None
        return CacheHit(found[0], "semantic", found[1])

    async def store(self, messages: Sequence[Message], scope: str, body: dict[str, Any]) -> None:
        question = messages[-1].content
        vector = None
        if self._embedder is not None and len(messages) == 1:
            vector = await anyio.to_thread.run_sync(self._embedder.embed_query, normalize(question))
        self._store.put(
            self._key(messages, scope), scope, question, figures(question), vector, body,
            timedelta(hours=self._config.ttl_hours),
        )  # fmt: skip

    @staticmethod
    def _key(messages: Sequence[Message], scope: str) -> str:
        turns = [(m.role.value, normalize(m.content)) for m in messages]
        return cache_scope(scope=scope, turns=turns)
