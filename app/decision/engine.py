"""Decision engine (spec section 5): Gate 2 dimension scores -> one decision and one code."""
from dataclasses import dataclass, field

from app.core.config import settings
from app.gates.gate2 import Gate2Result


@dataclass
class Verdict:
    decision: str                       # APPROVED | HUMAN_REVIEW | REJECTED
    composite_score: float | None
    rejection_code: str | None = None
    hard_reject: bool = False
    flags: list[str] = field(default_factory=list)


def composite_score(dimensions: dict[str, float | None]) -> float | None:
    """Spec 3.2 weighted sum; None while any dimension has no detector yet."""
    if any(value is None for value in dimensions.values()):
        return None
    weights = settings.gate2.weights
    return round(sum(dimensions[d] * weights[d] for d in weights), 1)


def pick_code(codes: list[str]) -> str:
    """Spec 5.2: exactly one primary code, by the priority order in settings."""
    order = settings.gate2.code_priority
    return min(codes, key=order.index)


def decide(result: Gate2Result) -> Verdict:
    policy = settings.gate2
    composite = composite_score(result.dimensions)
    flags = list(result.fired)

    if result.hard_signals:
        code = pick_code([policy.signals[s].code for s in result.hard_signals])
        return Verdict("REJECTED", composite, code, hard_reject=True, flags=flags)

    if composite is None:
        return Verdict("HUMAN_REVIEW", None, flags=flags + ["gate2_incomplete"])

    if composite < policy.thresholds.review:
        codes = [policy.signals[s].code for s in result.fired]
        code = pick_code(codes) if codes else policy.low_score_code
        return Verdict("REJECTED", composite, code, flags=flags)

    low = [d for d, bar in policy.review_below.items() if result.dimensions[d] < bar]
    flags += [f"low_{d}" for d in low]
    if composite < policy.thresholds.approve or result.review_signals or low:
        return Verdict("HUMAN_REVIEW", composite, flags=flags)
    return Verdict("APPROVED", composite, flags=flags)
