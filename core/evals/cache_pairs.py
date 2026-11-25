"""Choosing the semantic cache threshold from evidence instead of a guess.

The pack lists pairs of questions with whether one's answer may be reused for the other.
For each threshold this reports how many good reuses would happen (hits) and how many
wrong answers would be served (false hits). A false hit gives a user someone else's answer
to a different question, so the useful threshold is the lowest one with none.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.cache import figures, normalize
from core.evals.cases import EvalError
from core.rag import Embedder


@dataclass(frozen=True, slots=True)
class Pair:
    a: str
    b: str
    same: bool  # True when b may be answered with a's answer


@dataclass(frozen=True, slots=True)
class PairScore:
    pair: Pair
    similarity: float
    figures_match: bool


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    threshold: float
    hits: int  # same-answer pairs the cache would serve
    false_hits: int  # different-answer pairs the cache would wrongly serve


def load_pairs(path: Path) -> list[Pair]:
    if not path.is_file():
        raise EvalError(f"no cache pairs at {path}")
    pairs = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            raw = json.loads(line)
            pairs.append(Pair(str(raw["a"]), str(raw["b"]), bool(raw["same"])))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise EvalError(f"{path}:{number}: {exc}") from exc
    return pairs


def score_pairs(pairs: Sequence[Pair], embedder: Embedder) -> list[PairScore]:
    scores = []
    for pair in pairs:
        a = embedder.embed_query(normalize(pair.a))
        b = embedder.embed_query(normalize(pair.b))
        similarity = float(np.dot(a, b))
        scores.append(PairScore(pair, similarity, figures(pair.a) == figures(pair.b)))
    return scores


def sweep(scores: Sequence[PairScore], thresholds: Sequence[float]) -> list[ThresholdResult]:
    results = []
    for threshold in thresholds:
        served = [s for s in scores if s.figures_match and s.similarity >= threshold]
        results.append(
            ThresholdResult(
                threshold,
                hits=sum(s.pair.same for s in served),
                false_hits=sum(not s.pair.same for s in served),
            )
        )
    return results
