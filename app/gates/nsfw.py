"""D2 human nudity: the NSFW model's probability -> a gate2 signal (settings.gate2.nsfw)."""
from app.core.config import settings


def nsfw_signals(probability: float) -> set[str]:
    """The highest level the probability reaches, or nothing."""
    reached = [(level, signal) for signal, level in settings.gate2.nsfw.levels.items() if probability >= level]
    return {max(reached)[1]} if reached else set()
