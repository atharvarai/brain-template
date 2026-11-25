"""Layer 03, retrieval: load, chunk, embed, store, search and cite company documents."""

from core.rag.chunker import Chunker, MarkdownChunker
from core.rag.context import NO_DOCUMENTS, Source, format_documents, to_sources
from core.rag.embedder import Embedder, FastEmbedEmbedder, HashingEmbedder, build_embedder
from core.rag.errors import RagError
from core.rag.factory import RagComponents, build_rag
from core.rag.ingest import Ingestor, IngestReport
from core.rag.loaders import Loader, MarkdownFolderLoader
from core.rag.reranker import (
    FastEmbedReranker,
    OverlapReranker,
    Reranker,
    RerankingRetriever,
    build_reranker,
)
from core.rag.retriever import HybridRetriever, NullRetriever, Retriever, reciprocal_rank_fusion
from core.rag.store import ChunkStore, SqliteChunkStore
from core.rag.types import Chunk, Document, ScoredChunk

__all__ = [
    "NO_DOCUMENTS",
    "Chunk",
    "ChunkStore",
    "Chunker",
    "Document",
    "Embedder",
    "FastEmbedEmbedder",
    "FastEmbedReranker",
    "HashingEmbedder",
    "HybridRetriever",
    "IngestReport",
    "Ingestor",
    "Loader",
    "MarkdownChunker",
    "MarkdownFolderLoader",
    "NullRetriever",
    "OverlapReranker",
    "RagComponents",
    "RagError",
    "Reranker",
    "RerankingRetriever",
    "Retriever",
    "ScoredChunk",
    "Source",
    "SqliteChunkStore",
    "build_embedder",
    "build_rag",
    "build_reranker",
    "format_documents",
    "reciprocal_rank_fusion",
    "to_sources",
]
