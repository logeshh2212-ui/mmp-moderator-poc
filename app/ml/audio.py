"""Spec 6.2 audio models: AudioSet sound events (AST) and speech to text (Whisper)."""
import logging

import app.core.config  # noqa: F401  loads .env (HF_HOME, offline) before transformers
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import ASTFeatureExtractor, ASTForAudioClassification, pipeline

from app.core.config import MODEL_IGNORE_PATTERNS, settings


logger = logging.getLogger("mmp.ml")


def local_folder(name: str) -> str:
    """The model's folder in the D: cache (never downloads), as scripts.download_models left it."""
    spec = settings.ml.models[name]
    return snapshot_download(spec.repo, revision=spec.revision, local_files_only=True,
                             ignore_patterns=MODEL_IGNORE_PATTERNS)


class AudioEvents:
    """AST fine-tuned on AudioSet: a 0-1 score for each of 527 sound labels."""

    def __init__(self, name: str):
        self.name = name
        self.extractor = None
        self.model = None

    def load(self) -> None:
        folder = local_folder(self.name)
        self.extractor = ASTFeatureExtractor.from_pretrained(folder)
        self.model = ASTForAudioClassification.from_pretrained(folder).eval().to(settings.ml.device)
        logger.info("%s loaded (%d sound labels)", self.name, len(self.model.config.id2label))

    def warmup(self) -> None:
        self.scores(np.zeros(settings.gate2.audio.sample_rate, dtype=np.float32))

    @torch.inference_mode()
    def scores(self, wave: np.ndarray) -> dict[str, float]:
        """Highest score per label over chunk_seconds pieces (a 2 s yelp in a 60 s video counts)."""
        cfg = settings.gate2.audio
        size = int(cfg.chunk_seconds * cfg.sample_rate)
        chunks = [wave[i:i + size] for i in range(0, len(wave), size)]
        # The spectrogram needs at least ~1 s: drop a tiny tail piece, or pad a tiny whole clip.
        if len(chunks) > 1 and len(chunks[-1]) < cfg.sample_rate:
            chunks.pop()
        chunks = [np.pad(c, (0, max(0, cfg.sample_rate - len(c)))) for c in chunks] or [np.zeros(cfg.sample_rate, np.float32)]
        inputs = self.extractor(chunks, sampling_rate=cfg.sample_rate, return_tensors="pt")
        probs = torch.sigmoid(self.model(**inputs.to(settings.ml.device)).logits).max(dim=0).values
        labels = self.model.config.id2label
        return {labels[i]: float(p) for i, p in enumerate(probs)}


class SpeechToText:
    """Whisper: what is said in the video, any of 99 languages (Hindi included)."""

    def __init__(self, name: str):
        self.name = name
        self.pipe = None

    def load(self) -> None:
        self.pipe = pipeline("automatic-speech-recognition", model=local_folder(self.name),
                             device=settings.ml.device, chunk_length_s=30)
        logger.info("%s loaded (speech to text)", self.name)

    def warmup(self) -> None:
        self.transcribe(np.zeros(settings.gate2.audio.sample_rate, dtype=np.float32))

    def transcribe(self, wave: np.ndarray) -> str:
        rate = settings.gate2.audio.sample_rate
        out = self.pipe({"raw": wave, "sampling_rate": rate}, generate_kwargs={"task": "transcribe"})
        return out["text"].strip()
