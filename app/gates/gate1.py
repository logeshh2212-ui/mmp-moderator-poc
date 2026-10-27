from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from app.core.config import settings


Outcome = Literal["pass", "borderline", "fail"]


@dataclass
class FrameDetections:
    """What the detector branch found in one picture (filled by app.gates.detection)."""

    det_max: float             # best animal-box score 0-1 (0 = no animal box at all)
    det_label: str | None      # the detector's name for that box, e.g. "snail"
    species: list[str] = field(default_factory=list)  # final species names, best box first
    species_confidence: float = 0.0   # BioCLIP's probability for species[0] (0 = detector's name)
    lookalike: str | None = None      # best non-animal box, e.g. "statue (0.62)", for the log
    box: tuple[float, float, float, float] | None = None  # best animal box, detector picture pixels (D5)


@dataclass
class FrameVerdict:
    pet_score: float      # 0-100, the fused score
    threshold: float      # 75, or 60 for drawings / insects
    illustrated: bool
    insect: bool
    wild: bool
    best_animal: str      # SigLIP's best idea, e.g. "cats/a cat"
    best_no_pet: str
    animal_similarity: float   # raw SigLIP similarity of best_animal
    no_pet_similarity: float   # raw SigLIP similarity of best_no_pet
    siglip_score: float        # SigLIP's own 0-100 share
    det_max: float | None = None   # detector's best box score (None = detector not used)
    det_label: str | None = None
    species: list[str] = field(default_factory=list)
    lookalike: str | None = None


@dataclass
class Gate1Result:
    pet_score: int
    outcome: Outcome
    threshold: float
    wild: bool
    pass_fraction: float  # share of frames that passed (1.0 / 0.0 for an image)
    frames: list[FrameVerdict]
    species: list[str] = field(default_factory=list)  # for species_detected


def gate1_prompts() -> dict[str, str]:
    """All sentences SigLIP embeds at startup. Label = 'group|idea|wording number'."""
    g = settings.gate1
    prompts: dict[str, str] = {}
    for category, ideas in g.animals.items():
        for idea in ideas:
            for i, template in enumerate(g.templates):
                prompts[f"animal|{category}/{idea}|{i}"] = template.format(idea)
    for idea in g.no_pet:
        for i, template in enumerate(g.templates):
            prompts[f"no_pet|{idea}|{i}"] = template.format(idea)
    for group in ("photo", "illustration", "wild", "domestic"):
        for sentence in getattr(g, group):
            prompts[f"{group}|{sentence}|0"] = sentence
    return prompts


def idea_scores(sims: np.ndarray, labels: list[str]) -> dict[str, tuple[list[str], np.ndarray]]:
    """Average the wordings of each idea: {group: (idea names, matrix pictures x ideas)}."""
    groups: dict[str, dict[str, list[int]]] = {}
    for column, label in enumerate(labels):
        group, idea, _ = label.split("|")
        groups.setdefault(group, {}).setdefault(idea, []).append(column)
    return {
        group: (list(ideas), np.stack([sims[:, cols].mean(axis=1) for cols in ideas.values()], axis=1))
        for group, ideas in groups.items()
    }


def chance(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Chance that 'a' fits better than 'b' (two-way softmax of scaled similarities)."""
    return 1.0 / (1.0 + np.exp(-k * (a - b)))


def detector_score(det_max: float) -> float:
    """Detector box score 0-1 -> 0-100 between gate1.detector.score_floor and score_full."""
    d = settings.gate1.detector
    return float(np.clip((det_max - d.score_floor) / (d.score_full - d.score_floor), 0.0, 1.0) * 100)


def fuse(siglip_score: float, det_max: float) -> float:
    """pet_score from both branches. The detector decides 'is an animal visible', so an
    accessory-only photo (SigLIP high, no animal box) fails; SigLIP adds the scene view."""
    f = settings.gate1.fusion
    return f.detector_weight * detector_score(det_max) + f.siglip_weight * siglip_score


def score_frames(
    sims: np.ndarray,
    labels: list[str],
    scale: float,
    detections: list[FrameDetections] | None = None,
) -> list[FrameVerdict]:
    g = settings.gate1
    t = g.thresholds
    k = scale * g.score_scale
    s = idea_scores(sims, labels)

    animal_names, animal = s["animal"]
    no_pet_names, no_pet = s["no_pet"]
    # SigLIP score = share of ALL animal ideas among all ideas (softmax). Every matching animal
    # adds to the share, so a photo with a dog and a cat scores higher, not lower (spec 2.3).
    logits = np.concatenate([animal, no_pet], axis=1) * k
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    siglip = weights[:, : animal.shape[1]].sum(axis=1) / weights.sum(axis=1) * 100
    illustrated = chance(s["illustration"][1].max(axis=1), s["photo"][1].max(axis=1), k) > 0.5
    wild = chance(s["wild"][1].max(axis=1), s["domestic"][1].max(axis=1), k) > 0.5
    best_animal = animal.argmax(axis=1)
    best_no_pet = no_pet.argmax(axis=1)

    frames = []
    for i in range(sims.shape[0]):
        name = animal_names[best_animal[i]]
        category = name.split("/")[0]
        det = detections[i] if detections is not None else None
        if det is not None and det.species and det.species_confidence >= g.species_category_min_confidence:
            category = g.species[det.species[0]]  # a confident species model knows better than SigLIP
        pet = fuse(float(siglip[i]), det.det_max) if det is not None else float(siglip[i])

        insect = category in g.insect_categories
        if illustrated[i]:
            threshold = t.pass_illustrated
        elif insect:
            threshold = t.pass_insect
        else:
            threshold = t.pass_photo
        frames.append(FrameVerdict(
            pet_score=pet,
            threshold=threshold,
            illustrated=bool(illustrated[i]),
            insect=insect,
            wild=(bool(wild[i]) and category in g.wild_check_categories)
            or category in g.always_review_categories,
            best_animal=name,
            best_no_pet=no_pet_names[best_no_pet[i]],
            animal_similarity=float(animal[i].max()),
            no_pet_similarity=float(no_pet[i].max()),
            siglip_score=float(siglip[i]),
            det_max=det.det_max if det is not None else None,
            det_label=det.det_label if det is not None else None,
            species=list(det.species) if det is not None else [],
            lookalike=det.lookalike if det is not None else None,
        ))
    return frames


def outcome_for(score: float, threshold: float) -> Outcome:
    if score >= threshold:
        return "pass"
    if score >= settings.gate1.thresholds.fail_below:
        return "borderline"
    return "fail"


def run_gate1(
    sims: np.ndarray,
    labels: list[str],
    scale: float,
    is_video: bool,
    detections: list[FrameDetections] | None = None,
) -> Gate1Result:
    """Spec section 2 (images) and 6.1 (videos: pet in >= 30% of sampled frames)."""
    frames = score_frames(sims, labels, scale, detections)
    best = max(frames, key=lambda f: f.pet_score)

    if not is_video:
        outcome = outcome_for(best.pet_score, best.threshold)
        fraction = 1.0 if outcome == "pass" else 0.0
        wild = best.wild
        judged = frames
    else:
        passed = [f for f in frames if f.pet_score >= f.threshold]
        fraction = len(passed) / len(frames)
        if fraction >= settings.gate1.video_min_pass_fraction:
            outcome = "pass"
        elif best.pet_score >= settings.gate1.thresholds.fail_below:
            outcome = "borderline"  # a pet is seen, but in too few frames
        else:
            outcome = "fail"
        judged = passed or frames
        wild = sum(f.wild for f in judged) / len(judged) > 0.5

    species = [] if outcome == "fail" else list(dict.fromkeys(s for f in judged for s in f.species))
    return Gate1Result(
        pet_score=round(best.pet_score),
        outcome=outcome,
        threshold=best.threshold,
        wild=wild and outcome != "fail",
        pass_fraction=round(fraction, 3),
        frames=frames,
        species=species,
    )
