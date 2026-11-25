"""Prompt-leak guard: an answer that copies a run of the system prompt's wording.

The persona and the company description are meant to be repeated, so only the rest of the
system prompt (the rules) is protected.
"""

import re

_WORD = re.compile(r"[a-z0-9']+")
_PERSONA = re.compile(r"<persona>.*?</persona>", re.DOTALL)


def _shingles(text: str, size: int) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


class PromptLeakGuard:
    def __init__(self, system_prompt: str, min_words: int = 8) -> None:
        protected = _PERSONA.sub(" ", system_prompt).split("\n", 1)[-1]  # drop the intro line
        self._size = min_words
        self._protected = _shingles(protected, min_words)

    def leaks(self, answer: str) -> bool:
        return bool(self._protected & _shingles(answer, self._size))
