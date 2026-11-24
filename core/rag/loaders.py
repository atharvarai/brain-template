"""Loaders turn a company's files into ``Document`` objects."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import yaml

from core.rag.errors import RagError
from core.rag.types import Document

_FRONT_MATTER_FENCE = "---"


class Loader(Protocol):
    def load(self) -> Iterator[Document]: ...


class MarkdownFolderLoader:
    """Loads every ``*.md`` file under a folder.

    Optional YAML front matter sets ``title`` and ``labels``::

        ---
        title: Refund Policy
        labels: [internal]
        ---
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def load(self) -> Iterator[Document]:
        if not self._directory.is_dir():
            raise RagError(f"documents folder not found: {self._directory}")
        for path in sorted(self._directory.rglob("*.md")):
            yield self._load_file(path)

    def _load_file(self, path: Path) -> Document:
        meta, body = _split_front_matter(path, path.read_text(encoding="utf-8"))
        labels = meta.get("labels") or []
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise RagError(f"{path}: 'labels' must be a list of strings")
        title = meta.get("title") or _first_heading(body) or path.stem.replace("-", " ").title()
        return Document(
            source=path.relative_to(self._directory).as_posix(),
            title=str(title),
            text=body.strip(),
            labels=frozenset(labels),
        )


def _split_front_matter(path: Path, text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONT_MATTER_FENCE:
        return {}, text
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == _FRONT_MATTER_FENCE)
    except StopIteration:
        raise RagError(f"{path}: front matter is not closed with '---'") from None
    try:
        meta = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError as exc:
        raise RagError(f"{path}: front matter is not valid YAML: {exc}") from exc
    if not isinstance(meta, dict):
        raise RagError(f"{path}: front matter must be a mapping")
    return meta, "\n".join(lines[end + 1 :])


def _first_heading(body: str) -> str | None:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None
