import logging

import app.core.config  # noqa: F401  loads .env before torch
import numpy as np
from PIL import Image

from app.core.config import settings


logger = logging.getLogger("mmp.ml")


class OcrReader:
    """EasyOCR (Apache-2.0): the text written on a picture, e.g. meme captions and overlays.

    Weights live in gate2.text.ocr.model_dir and are fetched by scripts.download_models;
    the server never downloads (download_enabled=False).
    """

    name = "ocr"

    def __init__(self):
        self.cfg = settings.gate2.text.ocr
        self.reader = None

    def load(self) -> None:
        import easyocr  # heavy import, only when OCR is switched on

        self.reader = easyocr.Reader(
            self.cfg.languages, gpu=settings.ml.device == "cuda",
            model_storage_directory=str(self.cfg.model_dir),
            user_network_directory=str(self.cfg.model_dir / "user_network"),  # else it creates one on C:
            download_enabled=False, verbose=False,
        )
        logger.info("ocr loaded (EasyOCR %s)", "+".join(self.cfg.languages))

    def warmup(self) -> None:
        self.read([Image.new("RGB", (64, 64), "white")])

    def read(self, images: list[Image.Image]) -> list[str]:
        """All text found on each picture with at least ocr.min_confidence, top to bottom."""
        texts = []
        for image in images:
            boxes = self.reader.readtext(np.asarray(image.convert("RGB")), detail=1, paragraph=False)
            kept = [(box[0][1], text) for box, text, conf in boxes if conf >= self.cfg.min_confidence]
            texts.append(" ".join(text for _, text in sorted(kept)))
        return texts
