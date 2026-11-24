"""Where chunks and their vectors live.

``SqliteChunkStore`` is the zero-install adapter for local use: vectors are compared by
brute force in numpy and keywords use SQLite's built-in FTS5 index. That is fine up to tens
of thousands of chunks. A production deployment implements the same port on a real vector
database and pushes the label filter into the query.
"""

import json
import re
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from core.rag.embedder import Vector
from core.rag.errors import RagError
from core.rag.types import Chunk, ScoredChunk

_WORD = re.compile(r"\w+")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    position INTEGER NOT NULL,
    title TEXT NOT NULL,
    heading TEXT NOT NULL,
    text TEXT NOT NULL,
    labels TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_source ON chunks (source);
-- porter stemming lets "refund" match "refunds" and "refunded"
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5 (body, tokenize = 'porter unicode61');
"""


class ChunkStore(Protocol):
    """Port for chunk storage and the two kinds of search.

    ``allowed_labels`` is the access filter: ``None`` means no restriction, otherwise a chunk
    is visible only if every one of its labels is allowed. Unlabelled chunks are public.
    """

    def bind_embedder(self, name: str, reset: bool = False) -> None:
        """Record which embedder built the index, or fail if a different one did.

        With ``reset`` the index is emptied first, whatever built it."""
        ...

    def document_hashes(self) -> dict[str, str]: ...

    def upsert_document(
        self, source: str, content_hash: str, chunks: Sequence[Chunk], vectors: Vector
    ) -> None: ...

    def delete_document(self, source: str) -> None: ...

    def count(self) -> int: ...

    def vector_search(
        self, vector: Vector, k: int, allowed_labels: frozenset[str] | None
    ) -> list[ScoredChunk]: ...

    def keyword_search(
        self, query: str, k: int, allowed_labels: frozenset[str] | None
    ) -> list[ScoredChunk]: ...


def _visible(labels: frozenset[str], allowed: frozenset[str] | None) -> bool:
    return allowed is None or labels <= allowed


class SqliteChunkStore:
    def __init__(self, path: Path | str) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared across worker threads, guarded by a lock.
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._cache: tuple[list[Chunk], Vector] | None = None
        with self._lock, self._db:
            self._db.executescript(_SCHEMA)

    def bind_embedder(self, name: str, reset: bool = False) -> None:
        with self._lock, self._db:
            row = self._db.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()
            if row is not None and row[0] != name and not reset:
                raise RagError(
                    f"the index was built with embedder '{row[0]}' but the configuration "
                    f"now says '{name}'. Vectors from different models cannot be "
                    "compared: rebuild the index (ingest with --rebuild)."
                )
            if reset:
                self._db.execute("DELETE FROM chunks")
                self._db.execute("DELETE FROM chunks_fts")
                self._cache = None
            self._db.execute("INSERT OR REPLACE INTO meta VALUES ('embedder', ?)", (name,))

    def document_hashes(self) -> dict[str, str]:
        with self._lock:
            rows = self._db.execute("SELECT DISTINCT source, content_hash FROM chunks")
            return dict(rows.fetchall())

    def upsert_document(
        self, source: str, content_hash: str, chunks: Sequence[Chunk], vectors: Vector
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("one vector is needed per chunk")
        with self._lock, self._db:
            self._delete(source)
            for chunk, vector in zip(chunks, vectors, strict=True):
                cursor = self._db.execute(
                    "INSERT INTO chunks (source, position, title, heading, text, labels,"
                    " content_hash, vector) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk.source,
                        chunk.position,
                        chunk.title,
                        chunk.heading,
                        chunk.text,
                        json.dumps(sorted(chunk.labels)),
                        content_hash,
                        np.asarray(vector, dtype=np.float32).tobytes(),
                    ),
                )
                self._db.execute(
                    "INSERT INTO chunks_fts (rowid, body) VALUES (?, ?)",
                    (cursor.lastrowid, chunk.searchable_text),
                )
            self._cache = None

    def delete_document(self, source: str) -> None:
        with self._lock, self._db:
            self._delete(source)
            self._cache = None

    def count(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def vector_search(
        self, vector: Vector, k: int, allowed_labels: frozenset[str] | None
    ) -> list[ScoredChunk]:
        chunks, matrix = self._load_all()
        if not chunks:
            return []
        scores = matrix @ np.asarray(vector, dtype=np.float32)  # cosine: rows are unit length
        ranked = [
            ScoredChunk(chunks[index], float(scores[index]))
            for index in np.argsort(-scores)
            if _visible(chunks[index].labels, allowed_labels)
        ]
        return ranked[:k]

    def keyword_search(
        self, query: str, k: int, allowed_labels: frozenset[str] | None
    ) -> list[ScoredChunk]:
        words = _WORD.findall(query.lower())
        if not words:
            return []
        # Quote every word so user text can never be read as FTS5 query syntax.
        match = " OR ".join(f'"{word}"' for word in dict.fromkeys(words))
        with self._lock:
            rows = self._db.execute(
                "SELECT c.source, c.position, c.title, c.heading, c.text, c.labels,"
                " bm25(chunks_fts) FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid"
                " WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts)",
                (match,),
            ).fetchall()
        ranked = [ScoredChunk(_chunk(row), -float(row[6])) for row in rows]  # bm25: lower=better
        return [hit for hit in ranked if _visible(hit.chunk.labels, allowed_labels)][:k]

    def _delete(self, source: str) -> None:
        ids = [
            row[0] for row in self._db.execute("SELECT id FROM chunks WHERE source = ?", (source,))
        ]
        self._db.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(i,) for i in ids])
        self._db.execute("DELETE FROM chunks WHERE source = ?", (source,))

    def _load_all(self) -> tuple[list[Chunk], Vector]:
        with self._lock:
            if self._cache is None:
                rows = self._db.execute(
                    "SELECT source, position, title, heading, text, labels, vector"
                    " FROM chunks ORDER BY id"
                ).fetchall()
                chunks = [_chunk(row) for row in rows]
                vectors = [np.frombuffer(row[6], dtype=np.float32) for row in rows]
                matrix = np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)
                self._cache = (chunks, matrix)
            return self._cache


def _chunk(row: Sequence[object]) -> Chunk:
    source, position, title, heading, text, labels = row[:6]
    return Chunk(
        source=str(source),
        position=int(str(position)),
        title=str(title),
        heading=str(heading),
        text=str(text),
        labels=frozenset(json.loads(str(labels))),
    )
