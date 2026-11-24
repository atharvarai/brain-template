"""Ingestion: load, chunk, embed and store a company's documents."""

import logging
from dataclasses import dataclass

from core.rag.chunker import Chunker
from core.rag.embedder import Embedder
from core.rag.errors import RagError
from core.rag.loaders import Loader
from core.rag.store import ChunkStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestReport:
    added: int
    updated: int
    unchanged: int
    removed: int
    chunks: int  # total chunks in the index afterwards


class Ingestor:
    """Keeps the index in step with the documents.

    A run is incremental: unchanged documents are skipped (by content hash), changed ones are
    re-embedded, and documents that no longer exist are removed, so stale text can never be
    retrieved after a file is deleted or edited.
    """

    def __init__(
        self, loader: Loader, chunker: Chunker, embedder: Embedder, store: ChunkStore
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    def run(self, rebuild: bool = False) -> IngestReport:
        self._store.bind_embedder(self._embedder.name, reset=rebuild)
        indexed = self._store.document_hashes()
        seen: set[str] = set()
        added = updated = unchanged = 0

        for document in self._loader.load():
            if document.source in seen:
                raise RagError(f"two documents share the source id '{document.source}'")
            seen.add(document.source)
            if indexed.get(document.source) == document.content_hash:
                unchanged += 1
                continue
            chunks = self._chunker.split(document)
            if not chunks:
                logger.warning("skipping empty document: %s", document.source)
                continue
            vectors = self._embedder.embed_documents([chunk.searchable_text for chunk in chunks])
            self._store.upsert_document(document.source, document.content_hash, chunks, vectors)
            if document.source in indexed:
                updated += 1
            else:
                added += 1

        removed = sorted(set(indexed) - seen)
        for source in removed:
            self._store.delete_document(source)
        return IngestReport(added, updated, unchanged, len(removed), self._store.count())
