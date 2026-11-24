"""Chunkers split a document into retrievable pieces."""

import re
from typing import Protocol

from core.rag.types import Chunk, Document

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class Chunker(Protocol):
    def split(self, document: Document) -> list[Chunk]: ...


class MarkdownChunker:
    """Splits on Markdown headings first, then packs paragraphs up to ``max_chars``.

    Following the document's own structure keeps each chunk about one thing, and every chunk
    carries its heading path, so no overlap between chunks is needed.
    """

    def __init__(self, max_chars: int = 1200) -> None:
        if max_chars < 100:
            raise ValueError("max_chars must be at least 100")
        self._max_chars = max_chars

    def split(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for heading, body in self._sections(document):
            for piece in self._pack(body):
                chunks.append(
                    Chunk(
                        source=document.source,
                        position=len(chunks),
                        title=document.title,
                        heading=heading,
                        text=piece,
                        labels=document.labels,
                    )
                )
        return chunks

    def _sections(self, document: Document) -> list[tuple[str, str]]:
        """(heading path, body) pairs. The top-level title is left out of the path."""
        sections: list[tuple[str, str]] = []
        path: list[tuple[int, str]] = []
        lines: list[str] = []
        in_code_block = False

        def flush() -> None:
            body = "\n".join(lines).strip()
            if body:
                heading = " > ".join(text for level, text in path if level > 1)
                sections.append((heading, body))
            lines.clear()

        for line in document.text.splitlines():
            if line.lstrip().startswith("```"):
                in_code_block = not in_code_block
            match = None if in_code_block else _HEADING.match(line)
            if match is None:
                lines.append(line)
                continue
            flush()
            level = len(match.group(1))
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, match.group(2)))
        flush()
        return sections

    def _pack(self, body: str) -> list[str]:
        pieces: list[str] = []
        current = ""
        for paragraph in self._paragraphs(body):
            candidate = f"{current}\n\n{paragraph}" if current else paragraph
            if len(candidate) <= self._max_chars:
                current = candidate
                continue
            if current:
                pieces.append(current)
            current = paragraph
        if current:
            pieces.append(current)
        return pieces

    def _paragraphs(self, body: str) -> list[str]:
        """Paragraphs, with any paragraph longer than the limit split at sentence ends."""
        result: list[str] = []
        for paragraph in re.split(r"\n\s*\n", body):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) <= self._max_chars:
                result.append(paragraph)
                continue
            current = ""
            for sentence in _SENTENCE_END.split(paragraph):
                while len(sentence) > self._max_chars:  # no sentence end to split on
                    result.append(sentence[: self._max_chars])
                    sentence = sentence[self._max_chars :]
                candidate = f"{current} {sentence}" if current else sentence
                if len(candidate) <= self._max_chars:
                    current = candidate
                else:
                    result.append(current)
                    current = sentence
            if current:
                result.append(current)
        return result
