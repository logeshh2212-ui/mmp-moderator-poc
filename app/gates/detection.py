from PIL import Image

from app.core.config import settings
from app.gates.gate1 import FrameDetections


def detector_frame_indices(count: int, is_video: bool) -> list[int]:
    """Which prepared frames the (slow) detector looks at. Images: the one frame.
    Videos: up to gate1.detector.video_frames, evenly spaced, always first and last."""
    if not is_video:
        return [0]
    n = min(count, settings.gate1.detector.video_frames)
    if n <= 1:
        return [0]
    return sorted({round(i * (count - 1) / (n - 1)) for i in range(n)})


def _crop(img: Image.Image, xyxy: tuple[float, float, float, float]) -> Image.Image:
    x0, y0, x1, y1 = (int(round(v)) for v in xyxy)
    return img.crop((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)))


def detect_animals(pictures: list[Image.Image], detector, species_model=None) -> list[FrameDetections]:
    """Detector boxes per picture; the species model names the best few boxes."""
    cfg = settings.gate1.detector
    results = []
    sure = settings.gate1.species_min_confidence
    for img, all_boxes in zip(pictures, detector.detect(pictures)):
        # Look-alike boxes (statue, plush toy...) never count as an animal.
        boxes = [b for b in all_boxes if getattr(b, "animal", True)]
        lookalike = next(
            (b for b in all_boxes if not getattr(b, "animal", True) and not getattr(b, "item", False)),
            None,
        )
        lookalike_text = f"{lookalike.label} ({lookalike.score:.2f})" if lookalike else None
        items = {}  # Gate 2 (D4): best score per item query; boxes come best first
        objects = {}  # every box, by the name it fits best (shown in the response for analysis)
        for b in all_boxes:
            objects.setdefault(b.label, b.score)
            if getattr(b, "item", False):
                items.setdefault(b.label, b.score)
        if not boxes:
            results.append(FrameDetections(
                det_max=0.0, det_label=None, lookalike=lookalike_text, items=items, objects=objects,
            ))
            continue
        good = [b for b in boxes if b.score >= cfg.min_box_score][: cfg.max_species_boxes]
        names = [b.label for b in good]
        confidences = [0.0] * len(good)  # 0 = the name came from the detector, not BioCLIP
        if good and species_model is not None:
            named = species_model.classify([_crop(img, b.xyxy) for b in good])
            names = [name if conf >= sure else box.label for box, (name, conf) in zip(good, named)]
            confidences = [conf if conf >= sure else 0.0 for _, conf in named]
        results.append(FrameDetections(
            det_max=boxes[0].score,
            det_label=boxes[0].label,
            species=list(dict.fromkeys(names)),
            species_confidence=confidences[0] if confidences else 0.0,
            lookalike=lookalike_text,
            box=boxes[0].xyxy,
            items=items,
            objects=objects,
        ))
    return results
