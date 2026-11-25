"""Retrieval value objects. Nothing here knows what a document is about."""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Document:
    """One source file. ``labels`` is the generic access hook: who may read it is decided
    by the access layer, which maps a user's roles to the labels they are allowed to see."""

    source: str  # stable id, for example a path relative to the docs folder
    title: str
    text: str
    labels: frozenset[str] = frozenset()

    @property
    def content_hash(self) -> str:
        payload = "\x1f".join([self.title, ",".join(sorted(self.labels)), self.text])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable piece of a document."""

    source: str
    position: int  # order within the document
    title: str
    heading: str  # path of section headings, "" for text before the first heading
    text: str
    labels: frozenset[str] = frozenset()

    @property
    def id(self) -> str:
        """Stable within one index build: the document plus the chunk's place in it."""
        return f"{self.source}#{self.position}"

    @property
    def searchable_text(self) -> str:
        """Text that is embedded and indexed: the chunk with its title and heading in front,
        so a chunk like "Within 7 days." still knows it belongs to a refund policy."""
        context = " > ".join(part for part in (self.title, self.heading) if part)
        return f"{context}\n{self.text}"


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float  # higher is better; only comparable within one result list
