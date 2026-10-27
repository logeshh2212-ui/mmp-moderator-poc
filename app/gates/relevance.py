"""D6 context relevance (spec 4.6): what kind of post is this, from the SigLIP embedding
Gate 1 already computed (no extra model pass).

SigLIP compares each picture with a few sentences per kind of post (settings.gate2.relevance).
The most likely kind gives the spec band; how sure SigLIP is places the score inside it.
D6 never holds or rejects a post on its own (decided); it only moves the composite.
"""
from dataclasses import dataclass

import numpy as np

from app.core.config import settings
from app.gates.gate1 import idea_scores

GROUP = "relevance"


@dataclass
class RelevanceResult:
    score: float        # D6, 0-100
    kind: str           # e.g. personal_moment, meme, advert
    confidence: float   # SigLIP's share for that kind among all kinds, 0-1


def relevance_prompts() -> dict[str, str]:
    """Sentences SigLIP embeds at startup. Label = 'relevance|kind|sentence number'."""
    return {
        f"{GROUP}|{kind}|{i}": sentence
        for kind, cfg in settings.gate2.relevance.kinds.items()
        for i, sentence in enumerate(cfg.sentences)
    }


def score_relevance(sims: np.ndarray, labels: list[str], scale: float) -> list[RelevanceResult] | None:
    """One result per picture; None if the backbone was loaded without the relevance sentences."""
    groups = idea_scores(sims, labels)
    if GROUP not in groups:
        return None
    kinds, matrix = groups[GROUP]
    logits = matrix * scale
    shares = np.exp(logits - logits.max(axis=1, keepdims=True))
    shares /= shares.sum(axis=1, keepdims=True)

    results = []
    for row in shares:
        best = int(row.argmax())
        low, high = settings.gate2.relevance.kinds[kinds[best]].band
        score = low + float(row[best]) * (high - low)
        results.append(RelevanceResult(round(score, 1), kinds[best], float(row[best])))
    return results
