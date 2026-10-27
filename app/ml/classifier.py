import logging

import app.core.config  # noqa: F401  loads .env (HF_HOME, offline) before transformers
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from app.core.config import settings


logger = logging.getLogger("mmp.ml")


class ImageClassifier:
    """A Hugging Face image classifier reduced to one probability: that of `label`.

    Used for Gate 2 D2 human nudity (label nsfw) and the AI-generated label (label ai).
    """

    def __init__(self, name: str, label: str):
        self.name = name
        self.label = label
        self.spec = settings.ml.models[name]
        self.model = None
        self.processor = None
        self.index: int | None = None
        self.device = torch.device(settings.ml.device)
        self.dtype = getattr(torch, settings.ml.dtype)

    def load(self) -> None:
        self.processor = AutoImageProcessor.from_pretrained(self.spec.repo, revision=self.spec.revision)
        self.model = AutoModelForImageClassification.from_pretrained(
            self.spec.repo, revision=self.spec.revision, dtype=torch.float32
        ).eval().to(self.device, self.dtype)
        labels = {name: int(i) for i, name in self.model.config.id2label.items()}
        self.index = labels[self.label]
        logger.info("%s loaded (%s @ %s), label %s", self.name, self.spec.repo, self.spec.revision[:7], self.label)

    def warmup(self) -> None:
        self.probability([Image.new("RGB", (64, 64), "gray")])

    @torch.inference_mode()
    def probability(self, images: list[Image.Image]) -> list[float]:
        """Probability of `label` for each picture, in batches of ml.batch_size."""
        out = []
        size = settings.ml.batch_size
        for start in range(0, len(images), size):
            batch = self.processor(images=images[start : start + size], return_tensors="pt")
            pixels = batch["pixel_values"].to(self.device, self.dtype)
            out += self.model(pixel_values=pixels).logits.float().softmax(-1)[:, self.index].tolist()
        return out
