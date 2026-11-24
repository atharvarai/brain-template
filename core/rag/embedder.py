"""Embedders turn text into unit-length vectors.

Documents and queries must be embedded by the same model, so every embedder has a ``name``
that is stored with the index and checked on every start.
"""

import re
import zlib
from collections.abc import Sequence
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from core.config import EmbedderConfig, EmbedderKind
from core.rag.errors import RagError

Vector = NDArray[np.float32]
_WORD = re.compile(r"\w+")


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Identity of the model, for example ``fastembed:BAAI/bge-small-en-v1.5``."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        """Return an array of shape (len(texts), dimension), each row unit length."""
        ...

    def embed_query(self, text: str) -> Vector:
        """Return one unit-length vector of shape (dimension,)."""
        ...


def _normalise(matrix: Vector) -> Vector:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return cast(Vector, (matrix / np.where(norms == 0, 1.0, norms)).astype(np.float32))


class HashingEmbedder:
    """Bag-of-words hashing: deterministic, instant, no model download.

    It only matches shared words, so it is for tests and smoke runs, not for real retrieval.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return f"hashing:{self._dimensions}"

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        matrix = np.zeros((len(texts), self._dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD.findall(text.lower()):
                matrix[row, zlib.crc32(word.encode("utf-8")) % self._dimensions] += 1.0
        return _normalise(matrix)

    def embed_query(self, text: str) -> Vector:
        return cast(Vector, self.embed_documents([text])[0])


class FastEmbedEmbedder:
    """Local embedding model through fastembed (ONNX, no GPU or API key needed).

    The model is downloaded on first use and loaded lazily, so start-up stays fast.
    """

    def __init__(self, model: str) -> None:
        self._model_name = model
        self._model: Any = None

    @property
    def name(self) -> str:
        return f"fastembed:{self._model_name}"

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return _normalise(np.asarray(list(self._load().passage_embed(list(texts)))))

    def embed_query(self, text: str) -> Vector:
        return _normalise(np.asarray(next(iter(self._load().query_embed(text)))))

    def _load(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise RagError(f"the 'fastembed' package is not installed ({exc})") from exc
            self._model = TextEmbedding(self._model_name)
        return self._model


def build_embedder(config: EmbedderConfig) -> Embedder:
    match config.kind:
        case EmbedderKind.FASTEMBED:
            return FastEmbedEmbedder(config.model)
        case EmbedderKind.HASHING:
            return HashingEmbedder()
