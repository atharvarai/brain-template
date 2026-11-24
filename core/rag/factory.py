"""Assembles the retrieval layer for a domain pack."""

from dataclasses import dataclass
from pathlib import Path

from core.config import DomainPack
from core.rag.chunker import MarkdownChunker
from core.rag.embedder import build_embedder
from core.rag.errors import RagError
from core.rag.ingest import Ingestor
from core.rag.loaders import MarkdownFolderLoader
from core.rag.retriever import HybridRetriever
from core.rag.store import SqliteChunkStore


@dataclass(frozen=True, slots=True)
class RagComponents:
    store: SqliteChunkStore
    ingestor: Ingestor
    retriever: HybridRetriever
    top_k: int


def build_rag(pack: DomainPack, data_dir: Path) -> RagComponents | None:
    """Build retrieval for ``pack``, or return ``None`` if the pack has no ``rag`` section."""
    config = pack.config.rag
    if config is None or pack.docs_dir is None:
        return None
    if not pack.docs_dir.is_dir():
        raise RagError(f"documents folder not found: {pack.docs_dir}")

    embedder = build_embedder(config.embedder)
    store = SqliteChunkStore(data_dir / pack.root.name / "rag.sqlite3")
    ingestor = Ingestor(
        loader=MarkdownFolderLoader(pack.docs_dir),
        chunker=MarkdownChunker(config.chunk_chars),
        embedder=embedder,
        store=store,
    )
    return RagComponents(store, ingestor, HybridRetriever(embedder, store), config.top_k)
