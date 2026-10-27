"""AI-generated label (not in the spec; never changes the decision).

Metadata is certain when present (C2PA / IPTC, Stable Diffusion PNG text, generator names);
otherwise the AI-image model's probability decides against settings.ai_label.threshold.
"""
from dataclasses import dataclass, field
from statistics import median

from app.core.config import settings


@dataclass
class AiLabel:
    generated: bool | None          # None = nothing to go on (no hints and no model)
    score: float | None             # 0-100
    flags: list[str] = field(default_factory=list)


def ai_label(hints: list[str], probabilities: list[float] | None) -> AiLabel:
    """hints: metadata ai_hints; probabilities: the model's, one per picture (None = no model)."""
    if hints:
        return AiLabel(True, 100.0, ["ai_generated", "ai_metadata"])
    if not probabilities:
        return AiLabel(None, None)
    probability = median(probabilities)   # a video: one odd frame does not decide
    generated = probability >= settings.ai_label.threshold
    return AiLabel(generated, round(probability * 100, 1), ["ai_generated"] if generated else [])
