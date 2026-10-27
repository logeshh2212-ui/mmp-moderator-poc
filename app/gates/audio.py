"""Spec 6.2: audio of a video. Animal distress and violence sounds from the AudioSet scores,
abusive words in the speech from the shared word lists. Each costs D1 20 points; audio
alone never hard-rejects (settings.gate2.audio, the signals in gate2.signals)."""
from dataclasses import dataclass, field

from app.core.config import settings
from app.gates.text import text_hits


@dataclass
class AudioResult:
    signals: set[str] = field(default_factory=set)
    sounds: dict[str, float] = field(default_factory=dict)   # the listed sounds that counted
    speech_words: int = 0                                     # listed words heard (never the words)


def audio_signals(sound_scores: dict[str, float] | None, transcript: str | None) -> AudioResult:
    cfg = settings.gate2.audio
    result = AudioResult()
    for signal, group in cfg.sounds.items():
        found = {label: sound_scores[label] for label in group.labels
                 if sound_scores and sound_scores.get(label, 0.0) >= group.min_score}
        if found:
            result.signals.add(signal)
            result.sounds |= {k: round(v, 2) for k, v in found.items()}
    if transcript:
        result.speech_words = len(text_hits(transcript))
        if result.speech_words:
            result.signals.add(cfg.speech_signal)
    return result
