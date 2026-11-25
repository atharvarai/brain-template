"""Assembles the retrieval layer for a domain pack."""

from dataclasses import dataclass
from pathlib import Path

from core.config import DomainPack
from core.rag.chunker import MarkdownChunker
from core.rag.embedder import Embedder, build_embedder
from core.rag.errors import RagError
from core.rag.ingest import Ingestor
from core.rag.loaders import MarkdownFolderLoader
from core.rag.reranker import RerankingRetriever, build_reranker
from core.rag.retriever import HybridRetriever, Retriever
from core.rag.stopwords import STOPWORDS
from core.rag.store import SqliteChunkStore


@dataclass(frozen=True, slots=True)
class RagComponents:
    store: SqliteChunkStore
    ingestor: Ingestor
    retriever: Retriever
    top_k: int
    embedder: Embedder  # also used by the semantic answer cache


def build_rag(pack: DomainPack, data_dir: Path) -> RagComponents | None:
    """Build retrieval for ``pack``, or return ``None`` if the pack has no ``rag`` section."""
    config = pack.config.rag
    if config is None or pack.docs_dir is None:
        return None
    if not pack.docs_dir.is_dir():
        raise RagError(f"documents folder not found: {pack.docs_dir}")

    embedder = build_embedder(config.embedder)
    store = SqliteChunkStore(pack.state_dir(data_dir) / "rag.sqlite3", STOPWORDS[config.stopwords])
    ingestor = Ingestor(
        loader=MarkdownFolderLoader(pack.docs_dir),
        chunker=MarkdownChunker(config.chunk_chars),
        embedder=embedder,
        store=store,
    )
    retriever: Retriever = HybridRetriever(embedder, store)
    if config.reranker is not None:
        retriever = RerankingRetriever(
            retriever, build_reranker(config.reranker), config.reranker.candidates
        )
    return RagComponents(store, ingestor, retriever, config.top_k, embedder)
