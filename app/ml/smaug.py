"""Smaug AI Image Detector (abacusai/Smaug-AI-Image-Detector, Apache-2.0): probability that
an image is AI-generated.

The cropping below is copied from the model repo's smaug_image_detect.py (revision c7c9595,
read in full on 2026-10-27) so no code from the model cache is ever imported. It must match
training: shorter side to 576 px, then 528 x 528 crops at native resolution.
"""
import logging
import math

import app.core.config  # noqa: F401  loads .env (HF_HOME, offline) before huggingface_hub
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from sentence_transformers import SentenceTransformer

from app.core.config import MODEL_IGNORE_PATTERNS, settings


logger = logging.getLogger("mmp.ml")

CROP, SIDE, BUDGET = 528, 576, 140  # crop size, shorter side, vision tokens (as trained)


def views(img: Image.Image, crops: int) -> list[Image.Image]:
    """crops=3: start, centre and end of the long side; crops=1: centre only."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    if min(w, h) > SIDE:
        s = SIDE / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    elif min(w, h) < CROP:
        s = CROP / min(w, h)
        img = img.resize((math.ceil(w * s), math.ceil(h * s)), Image.BICUBIC)
    w, h = img.size
    cx, cy = (w - CROP) // 2, (h - CROP) // 2
    if crops == 1:
        return [img.crop((cx, cy, cx + CROP, cy + CROP))]
    if w >= h:
        return [img.crop((x, cy, x + CROP, cy + CROP)) for x in (0, cx, w - CROP)]
    return [img.crop((cx, y, cx + CROP, y + CROP)) for y in (0, cy, h - CROP)]


class SmaugDetector:
    def __init__(self, name: str):
        self.name = name
        self.spec = settings.ml.models[name]
        self.model: SentenceTransformer | None = None

    def load(self) -> None:
        folder = snapshot_download(self.spec.repo, revision=self.spec.revision, local_files_only=True,
                                   ignore_patterns=MODEL_IGNORE_PATTERNS)
        # bfloat16 only where the GPU has it (A100 / L4, not T4); never float16 (the model card
        # says it is unstable in float16)
        bf16 = settings.ml.device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
        dtype = torch.bfloat16 if bf16 else torch.float32
        self.model = SentenceTransformer(
            folder, device=settings.ml.device, local_files_only=True,
            config_kwargs={"audio_config": None}, model_kwargs={"torch_dtype": dtype},
        )
        processor = self.model[0].processor.image_processor
        processor.max_soft_tokens = processor.image_seq_length = BUDGET
        logger.info("%s loaded (%s @ %s)", self.name, self.spec.repo, self.spec.revision[:7])

    def warmup(self) -> None:
        self.probability([Image.new("RGB", (64, 64), "gray")])

    @torch.inference_mode()
    def probability(self, images: list[Image.Image]) -> list[float]:
        """Probability that each picture is AI-generated (crop logits averaged, then sigmoid)."""
        crops = settings.ai_label.crops
        flat = [{"image": view} for image in images for view in views(image, crops)]
        logits = self.model.encode(flat, batch_size=settings.ml.batch_size, convert_to_tensor=True)
        return torch.sigmoid(logits.float().view(len(images), crops).mean(1)).tolist()
