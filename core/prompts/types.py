"""Prompt value objects."""

import hashlib
from dataclasses import dataclass

from core.prompts.template import render


@dataclass(frozen=True, slots=True)
class PromptRef:
    """Identifies exactly which prompt text produced an answer.

    The checksum catches the classic mistake of editing a prompt file without bumping its
    version: the version stays the same but the checksum in the logs changes.
    """

    name: str
    version: int
    checksum: str


@dataclass(frozen=True, slots=True)
class Prompt:
    name: str
    version: int
    template: str

    @property
    def ref(self) -> PromptRef:
        digest = hashlib.sha256(self.template.encode("utf-8")).hexdigest()[:12]
        return PromptRef(self.name, self.version, digest)

    def render(self, **variables: str) -> str:
        return render(self.template, variables)


@dataclass(frozen=True, slots=True)
class ComposedPrompt:
    """Final prompt text plus the refs of every prompt file that went into it."""

    text: str
    refs: tuple[PromptRef, ...]
