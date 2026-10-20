import logging

import app.core.config  # noqa: F401  loads .env (HF_HOME, offline) before huggingface_hub
import open_clip
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from app.core.config import settings


logger = logging.getLogger("mmp.ml")


class BioclipClassifier:
    """BioCLIP 2: names the species in an animal box (trained on 200M+ photos of organisms).

    It always names *some* organism, so it only runs on boxes the detector already found.
    """

    def __init__(self, name: str, species: list[str]):
        self.name = name
        self.spec = settings.ml.models[name]
        self.species = list(species)
        self.model = None
        self.preprocess = None
        self.text_embeds: torch.Tensor | None = None
        self.device = torch.device(settings.ml.device)

    def load(self) -> None:
        folder = snapshot_download(
            self.spec.repo,
            revision=self.spec.revision,
            local_files_only=True,
            ignore_patterns=["*.bin", "*.pt"],  # we keep only the .safetensors weights
        )
        precision = "fp16" if settings.ml.dtype == "float16" else "fp32"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            f"local-dir:{folder}", precision=precision, device=self.device
        )
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(f"local-dir:{folder}")
        with torch.inference_mode():
            text = self.model.encode_text(tokenizer([f"a photo of {s}." for s in self.species]).to(self.device))
            self.text_embeds = torch.nn.functional.normalize(text.float(), dim=-1)
        logger.info("%s loaded (%s @ %s), %d species",
                    self.name, self.spec.repo, self.spec.revision[:7], len(self.species))

    def warmup(self) -> None:
        self.classify([Image.new("RGB", (64, 64), "gray")])

    @torch.inference_mode()
    def classify(self, crops: list[Image.Image]) -> list[tuple[str, float]]:
        """Best species and its probability for each crop."""
        if not crops:
            return []
        batch = torch.stack([self.preprocess(c.convert("RGB")) for c in crops]).to(self.device)
        if settings.ml.dtype == "float16":
            batch = batch.half()
        embeds = torch.nn.functional.normalize(self.model.encode_image(batch).float(), dim=-1)
        probs = (100 * embeds @ self.text_embeds.T).softmax(dim=-1)
        best, index = probs.max(dim=1)
        return [(self.species[int(i)], float(p)) for p, i in zip(best, index)]
