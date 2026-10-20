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
        from app.ml.siglip import SiglipEmbedder

        registry.register(SiglipEmbedder(settings.ml.backbone))
    registry.load_all()


def get_backbone():
    """The loaded SigLIP 2 model, or None if none is configured or it failed to load."""
    name = settings.ml.backbone
    if name and registry.status.get(name) == "ready":
        return registry.get(name)
    return None
