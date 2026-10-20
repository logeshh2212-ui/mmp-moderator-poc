import gc
import logging

import app.core.config  # noqa: F401  loads .env (HF_HOME, offline) before transformers
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from app.core.config import settings


logger = logging.getLogger("mmp.ml")

TEXT_MAX_LENGTH = 64  # SigLIP was trained with text padded to 64 tokens


class SiglipEmbedder:
    """SigLIP 2 image embeddings plus zero-shot probabilities for a fixed set of prompts.

    The prompts are embedded once in load(); the text half of the model is then deleted
    to save memory, so changing the prompts needs a restart.
    """

    def __init__(self, name: str, prompts: dict[str, str] | None = None):
        self.name = name
        self.spec = settings.ml.models[name]
        self.prompts = dict(settings.ml.prompts if prompts is None else prompts)
        self.labels = list(self.prompts)
        self.vision = None
        self.image_processor = None
        self.text_embeds: torch.Tensor | None = None
        self.scale = 1.0
        self.bias = 0.0
        self.dim: int | None = None
        self.device = torch.device(settings.ml.device)
        self.dtype = getattr(torch, settings.ml.dtype)
    def load(self) -> None:
        processor = AutoProcessor.from_pretrained(self.spec.repo, revision=self.spec.revision)
        model = AutoModel.from_pretrained(
            self.spec.repo, revision=self.spec.revision, dtype=torch.float32
        ).eval()

        with torch.inference_mode():
            tokens = processor.tokenizer(
                list(self.prompts.values()),
                padding="max_length",
                max_length=TEXT_MAX_LENGTH,
                truncation=True,
                return_tensors="pt",
            )
            text = model.get_text_features(**tokens).pooler_output
            self.text_embeds = torch.nn.functional.normalize(text, dim=-1)

        self.scale = model.logit_scale.detach().exp().item()
        self.bias = model.logit_bias.detach().item()
        self.dim = int(self.text_embeds.shape[-1])

        # Keep only the image half; the text half (and the tokenizer) is freed.
        self.vision = model.vision_model.to(self.device, self.dtype)
        self.image_processor = processor.image_processor
        del model, processor, tokens, text
        gc.collect()
        logger.info(
            "%s loaded (%s @ %s), %d-d, text half dropped",
            self.name, self.spec.repo, self.spec.revision[:7], self.dim,
        )

    def warmup(self) -> None:
        self.embed_images([Image.new("RGB", (64, 64), "gray")])

    @torch.inference_mode()
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """One normalised embedding per picture, in batches of ml.batch_size."""
        out = []
        size = settings.ml.batch_size
        for start in range(0, len(images), size):
            batch = self.image_processor(images=images[start : start + size], return_tensors="pt")
            pixels = batch["pixel_values"].to(self.device, self.dtype)
            pooled = self.vision(pixel_values=pixels).pooler_output.float()
            out.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())


        return torch.cat(out).numpy()

    def probabilities(self, image_embeds: np.ndarray) -> dict[str, list[float]]:
        """SigLIP's independent yes/no probability per prompt (sigmoid, not softmax)."""
        logits = torch.from_numpy(image_embeds) @ self.text_embeds.T * self.scale + self.bias
        probs = torch.sigmoid(logits)
        return {label: probs[:, i].tolist() for i, label in enumerate(self.labels)}
