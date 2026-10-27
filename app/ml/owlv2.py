import logging
from dataclasses import dataclass

import app.core.config  # noqa: F401  loads .env (HF_HOME, offline) before transformers
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from app.core.config import settings


logger = logging.getLogger("mmp.ml")


@dataclass(frozen=True)
class Box:
    score: float                            # 0-1, how sure OWLv2 is that this box is `label`
    label: str                              # a species ("snail") or a look-alike ("statue")
    xyxy: tuple[float, float, float, float]  # pixel corners in the picture that was searched
    animal: bool = True                     # False when the best name is a look-alike or an item
    item: bool = False                      # True when the best name is a Gate 2 item query (D4)


class Owlv2Detector:
    """OWLv2 open-vocabulary detector: finds boxes for animals named in words.

    Unlike SigLIP it looks for objects, so an empty cage or tank gives no animal box.
    The same pass also looks for Gate 2 items (D4: weapons, drugs, alcohol, tobacco).
    """

    def __init__(
        self, name: str, labels: list[str], lookalikes: list[str] | None = None,
        items: list[str] | None = None,
    ):
        self.name = name
        self.spec = settings.ml.models[name]
        self.animal_labels = list(labels)
        self.lookalikes = list(lookalikes or [])
        self.items = list(items or [])
        # one OWLv2 query each; every box takes the name it fits best
        self.labels = self.animal_labels + self.lookalikes + self.items
        self.processor = None
        self.model = None
        self.text_inputs = None
        self.device = torch.device(settings.ml.device)
        self.dtype = getattr(torch, settings.ml.dtype)

    def load(self) -> None:
        self.processor = Owlv2Processor.from_pretrained(self.spec.repo, revision=self.spec.revision)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            self.spec.repo, revision=self.spec.revision, dtype=torch.float32
        ).eval().to(self.device, self.dtype)
        queries = [[f"a photo of a {label}" for label in self.labels]]
        self.text_inputs = self.processor(text=queries, return_tensors="pt").to(self.device)
        logger.info("%s loaded (%s @ %s), %d animal names, %d look-alikes, %d items",
                    self.name, self.spec.repo, self.spec.revision[:7],
                    len(self.animal_labels), len(self.lookalikes), len(self.items))

    def warmup(self) -> None:
        self.detect([Image.new("RGB", (64, 64), "gray")])

    @torch.inference_mode()
    def detect(self, images: list[Image.Image]) -> list[list[Box]]:
        """Boxes per picture, best first, above gate1.detector.score_floor. A box whose best
        name is a look-alike (statue, plush toy...) comes back with animal=False; one whose
        best name is an item query with animal=False, item=True."""
        floor = settings.gate1.detector.score_floor
        results = []
        for img in images:
            pixels = self.processor(images=img, return_tensors="pt")["pixel_values"]
            out = self.model(
                pixel_values=pixels.to(self.device, self.dtype),
                input_ids=self.text_inputs["input_ids"],
                attention_mask=self.text_inputs["attention_mask"],
            )
            scores = out.logits[0].float().sigmoid()   # boxes x labels
            best, label_index = scores.max(dim=1)       # per box: its best label
            # OWLv2 pads the picture to a square; boxes are (cx, cy, w, h) of that square.
            side = max(img.size)
            boxes = []
            for i in torch.nonzero(best >= floor).flatten().tolist():
                cx, cy, w, h = out.pred_boxes[0][i].float().tolist()
                xyxy = (
                    max(0.0, (cx - w / 2) * side),
                    max(0.0, (cy - h / 2) * side),
                    min(float(img.width), (cx + w / 2) * side),
                    min(float(img.height), (cy + h / 2) * side),
                )
                label_number = int(label_index[i])
                boxes.append(Box(
                    float(best[i]), self.labels[label_number], xyxy,
                    animal=label_number < len(self.animal_labels),
                    item=label_number >= len(self.animal_labels) + len(self.lookalikes),
                ))
            results.append(sorted(boxes, key=lambda b: -b.score))
        return results
