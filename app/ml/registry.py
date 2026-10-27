import logging
from typing import Protocol

from app.core.config import settings


logger = logging.getLogger("mmp.ml")


class Model(Protocol):
    """Every model wrapper (SigLIP 2, RF-DETR, BioCLIP 2) has these."""

    name: str

    def load(self) -> None: ...      # read weights from D:\hf-cache into memory
    def warmup(self) -> None: ...    # one tiny input, so the first real upload is not slow


class ModelRegistry:
    """Models are loaded once at startup and kept in memory."""

    def __init__(self) -> None:
        self._models: dict[str, Model] = {}
        self.status: dict[str, str] = {}

    def register(self, model: Model) -> None:
        self._models[model.name] = model
        self.status[model.name] = "registered"

    def load_all(self) -> None:
        for name, model in self._models.items():
            try:
                model.load()
                if settings.ml.warmup:
                    model.warmup()
                self.status[name] = "ready"
                logger.info("model %s ready", name)
            except Exception as exc:
                self.status[name] = f"failed: {exc}"
                logger.error("model %s failed to load: %s", name, exc)

    def get(self, name: str) -> Model:
        if self.status.get(name) != "ready":
            raise RuntimeError(f"model {name} is not ready ({self.status.get(name, 'unknown')})")
        return self._models[name]

    def clear(self) -> None:
        self._models.clear()
        self.status.clear()


registry = ModelRegistry()


def load_models() -> None:
    from app.ml.runtime import configure_torch  # imports torch only when we really need it

    configure_torch()
    if settings.ml.backbone:
        from app.gates.gate1 import gate1_prompts
        from app.gates.relevance import relevance_prompts
        from app.gates.scene import scene_prompts
        from app.ml.siglip import SiglipEmbedder

        # The text half is dropped after these are embedded: every sentence is needed here.
        prompts = gate1_prompts() | relevance_prompts() | scene_prompts()
        registry.register(SiglipEmbedder(settings.ml.backbone, prompts=prompts))
    species = list(settings.gate1.species)
    if settings.ml.detector:
        from app.ml.owlv2 import Owlv2Detector

        registry.register(Owlv2Detector(
            settings.ml.detector, labels=species, lookalikes=settings.gate1.detector.lookalikes,
            items=list(settings.gate2.items.queries),
        ))
    if settings.ml.species_model:
        from app.ml.bioclip import BioclipClassifier

        registry.register(BioclipClassifier(settings.ml.species_model, species=species))
    if settings.ml.nsfw_model:
        from app.ml.classifier import ImageClassifier

        registry.register(ImageClassifier(settings.ml.nsfw_model, label=settings.gate2.nsfw.label))
    if settings.ml.ai_model:
        from app.ml.smaug import SmaugDetector

        registry.register(SmaugDetector(settings.ml.ai_model))
    if settings.ml.ocr:
        from app.ml.ocr import OcrReader

        registry.register(OcrReader())
    if settings.ml.audio_model or settings.ml.speech_model:
        from app.ml.audio import AudioEvents, SpeechToText

        if settings.ml.audio_model:
            registry.register(AudioEvents(settings.ml.audio_model))
        if settings.ml.speech_model:
            registry.register(SpeechToText(settings.ml.speech_model))
    registry.load_all()


def _ready(name: str | None):
    if name and registry.status.get(name) == "ready":
        return registry.get(name)
    return None


def get_backbone():
    """The loaded SigLIP 2 model, or None if none is configured or it failed to load."""
    return _ready(settings.ml.backbone)


def get_detector():
    """The loaded animal detector (OWLv2), or None."""
    return _ready(settings.ml.detector)


def get_species_model():
    """The loaded species model (BioCLIP 2), or None."""
    return _ready(settings.ml.species_model)


def get_nsfw_model():
    """The loaded NSFW classifier (Gate 2 D2), or None."""
    return _ready(settings.ml.nsfw_model)


def get_ai_model():
    """The loaded AI-generated image detector, or None."""
    return _ready(settings.ml.ai_model)


def get_ocr():
    """The loaded text reader (EasyOCR), or None."""
    return _ready("ocr" if settings.ml.ocr else None)


def get_audio_model():
    """The loaded sound-event model (AST / AudioSet), or None."""
    return _ready(settings.ml.audio_model)


def get_speech_model():
    """The loaded speech-to-text model (Whisper), or None."""
    return _ready(settings.ml.speech_model)
