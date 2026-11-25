"""Grounding: does the answer stick to what it was given?

The numbers check is cheap and catches the costly kind of hallucination: a price, a date, a
count or a percentage that appears in none of the documents, tool results or the question.
Figures the model worked out itself (half of a fee, say) are flagged too, so the default
action is to flag, not block.
"""

import re
from collections.abc import Iterable

_CITATION = re.compile(r"\[\d+\]")
# In the answer, a figure stands on its own (not part of a word like "v2" or "BK-0001's").
_ANSWER_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
# In the evidence, any digit run counts: "2026-11-27T19:00" holds 19 and 00 too.
_EVIDENCE_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str, pattern: re.Pattern[str] = _ANSWER_NUMBER) -> set[str]:
    found = set()
    for raw in pattern.findall(_CITATION.sub(" ", text)):
        value = raw.replace(",", "").rstrip(".")
        normal = value.lstrip("0") or "0"
        if "." in normal:
            normal = normal.rstrip("0").rstrip(".")
        if len(normal) >= 2:  # single digits are too common in prose to judge
            found.add(normal)
    return found


def ungrounded_numbers(answer: str, evidence: Iterable[str]) -> list[str]:
    """Figures in ``answer`` that occur in none of the ``evidence`` texts."""
    known: set[str] = set()
    for text in evidence:
        known |= _numbers(text, _EVIDENCE_NUMBER)
    return sorted(_numbers(answer) - known, key=lambda n: (len(n), n))
