"""Gate 2 scoring (spec sections 3, 4, 6.3 and 7): detector signals -> six dimension scores.

The Gate 2 detectors (steps 2b-2e) only report what they saw in each picture. Every
number - deductions, context exceptions, the video window - lives in settings.gate2.
"""
from dataclasses import dataclass, field
from statistics import median

from app.core.config import settings

SAFETY = ("D1_welfare", "D2_nsfw", "D3_violence", "D4_prohibited")


@dataclass
class FrameSignals:
    """What the Gate 2 detectors saw in one image or video frame."""

    signals: set[str] = field(default_factory=set)    # ids from gate2.signals
    contexts: set[str] = field(default_factory=set)   # ids from gate2.contexts
    quality: float | None = None                      # D5, 0-100 (step 2b)
    relevance: float | None = None                    # D6, 0-100 (step 2c)


@dataclass
class Gate2Result:
    dimensions: dict[str, float | None]   # D1-D6; D5 / D6 are None until their detector exists
    hard_signals: list[str]               # hard-reject signals seen in any frame
    review_signals: list[str]             # signals that always need a human
    fired: list[str]                      # every signal that took points off, or is hard


def signal_deduction(signal_id: str, contexts: set[str]) -> float:
    """Points one signal takes off its dimension, after the spec 7 context exceptions."""
    policy = settings.gate2
    signal = policy.signals[signal_id]
    if signal.hard:
        return signal.deduction  # a context never softens a hard reject
    found = [policy.contexts[c] for c in sorted(contexts)]
    overrides = [c.deductions[signal_id] for c in found if signal_id in c.deductions]
    deduction = min(overrides) if overrides else signal.deduction
    for context in found:
        deduction *= context.dimension_factor.get(signal.dimension, 1.0)
    return deduction


def needs_review(signal_id: str, deduction: float, contexts: set[str]) -> bool:
    policy = settings.gate2
    return (
        policy.signals[signal_id].review
        or deduction >= policy.review_from_deduction
        or any(signal_id in policy.contexts[c].review_on for c in contexts)
    )


def frame_safety(frame: FrameSignals) -> tuple[dict[str, float], list[str], list[str]]:
    """D1-D4 for one frame (100 = no risk, minimum 0), the signals that counted,
    and the ones among them that need a human."""
    off = dict.fromkeys(SAFETY, 0.0)
    counted, review = [], []
    for signal_id in sorted(frame.signals):
        signal = settings.gate2.signals[signal_id]
        deduction = signal_deduction(signal_id, frame.contexts)
        off[signal.dimension] += deduction
        if deduction > 0 or signal.hard:
            counted.append(signal_id)
            if not signal.hard and needs_review(signal_id, deduction, frame.contexts):
                review.append(signal_id)
    return {d: max(0.0, 100.0 - off[d]) for d in SAFETY}, counted, review


def worst_window(values: list[float], size: int) -> float:
    """Lowest average over any `size` consecutive frames (spec 6.3); all frames if fewer."""
    size = min(size, len(values))
    return min(sum(values[i:i + size]) / size for i in range(len(values) - size + 1))


def score_gate2(frames: list[FrameSignals]) -> Gate2Result:
    """One image = one frame. A video is judged by its worst moment, never its average."""
    per_frame = [frame_safety(frame) for frame in frames]
    window = settings.gate2.video_window_frames

    dimensions: dict[str, float | None] = {
        d: worst_window([scores[d] for scores, _, _ in per_frame], window) for d in SAFETY
    }
    # Quality and relevance describe the whole post, so one bad frame does not decide them.
    quality = [f.quality for f in frames if f.quality is not None]
    relevance = [f.relevance for f in frames if f.relevance is not None]
    dimensions["D5_quality"] = float(median(quality)) if quality else None
    dimensions["D6_relevance"] = float(median(relevance)) if relevance else None

    fired = sorted({s for _, counted, _ in per_frame for s in counted})
    hard = [s for s in fired if settings.gate2.signals[s].hard]
    review = sorted({s for _, _, needs_human in per_frame for s in needs_human})
    return Gate2Result(dimensions=dimensions, hard_signals=hard, review_signals=review, fired=fired)
