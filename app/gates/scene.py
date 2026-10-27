"""D1 welfare, D3 violence and the scene parts of D2, plus the spec 7 contexts, from the
SigLIP embedding Gate 1 already computed (no extra model pass).

Zero-shot: SigLIP's own yes-probability for each sentence in settings.gate2.scene. A harm
sentence counts when it reaches min_probability and fits the picture better than every
safe sentence (everyday look-alikes such as play fighting or a cone collar).
"""
from dataclasses import dataclass, field

import numpy as np

from app.core.config import settings

SAFE, HARM, CONTEXT = "scene_safe", "scene_harm", "scene_context"


@dataclass
class SceneResult:
    signals: set[str] = field(default_factory=set)       # ids from gate2.signals
    contexts: set[str] = field(default_factory=set)      # ids from gate2.contexts
    harm: dict[str, float] = field(default_factory=dict)  # harm idea -> best counted probability


def scene_prompts() -> dict[str, str]:
    """Sentences SigLIP embeds at startup. Label = 'group|idea|sentence number'."""
    scene = settings.gate2.scene
    groups = (
        (SAFE, scene.safe),
        (HARM, {idea: cfg.sentences for idea, cfg in scene.harm.items()}),
        (CONTEXT, scene.contexts),
    )
    return {
        f"{group}|{idea}|{i}": sentence
        for group, ideas in groups
        for idea, sentences in ideas.items()
        for i, sentence in enumerate(sentences)
    }


def columns(labels: list[str], group: str) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for column, label in enumerate(labels):
        name, idea, _ = label.split("|")
        if name == group:
            found.setdefault(idea, []).append(column)
    return found


def score_scene(sims: np.ndarray, labels: list[str], scale: float, bias: float) -> list[SceneResult] | None:
    """One result per picture; None if the backbone was loaded without the scene sentences."""
    safe, harm, contexts = columns(labels, SAFE), columns(labels, HARM), columns(labels, CONTEXT)
    if not safe or not harm:
        return None
    cfg = settings.gate2.scene
    probability = 1.0 / (1.0 + np.exp(-(sims * scale + bias)))
    best_safe = sims[:, [c for cols in safe.values() for c in cols]].max(axis=1)

    results = [SceneResult() for _ in range(len(sims))]
    for idea, cols in harm.items():
        counted = np.where(sims[:, cols] > best_safe[:, None], probability[:, cols], 0.0).max(axis=1)
        for result, p in zip(results, counted):
            if p >= cfg.min_probability:
                result.signals.update(cfg.harm[idea].signals)
                result.harm[idea] = round(float(p), 3)
    for name, cols in contexts.items():
        for result, p in zip(results, probability[:, cols].max(axis=1)):
            if p >= cfg.context_min_probability:
                result.contexts.add(name)
    return results
