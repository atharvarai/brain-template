"""Turns retrieved chunks into prompt text and into citable sources."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from core.prompts import wrap_untrusted
from core.rag.types import Chunk

NO_DOCUMENTS = "(no matching documents were found)"
_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True, slots=True)
class Source:
    """A retrieved chunk as the caller sees it. ``index`` is the number the model cites."""

    index: int
    source: str
    title: str
    section: str
    cited: bool = False


def format_documents(chunks: Sequence[Chunk]) -> str:
    """Numbered, tag-wrapped documents. Document text can never close its own wrapper."""
    if not chunks:
        return NO_DOCUMENTS
    return "\n".join(
        wrap_untrusted(
            "document", chunk.text, index=index, title=chunk.title, section=chunk.heading
        )
        for index, chunk in enumerate(chunks, start=1)
    )


def to_sources(chunks: Sequence[Chunk], answer: str) -> tuple[Source, ...]:
    """Sources for the response, each flagged with whether the answer actually cited it."""
    cited = {int(number) for number in _CITATION.findall(answer)}
    return tuple(
        Source(index, chunk.source, chunk.title, chunk.heading, cited=index in cited)
        for index, chunk in enumerate(chunks, start=1)
    )
