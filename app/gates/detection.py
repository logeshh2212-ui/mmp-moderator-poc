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
    for img, boxes in zip(pictures, detector.detect(pictures)):
        if not boxes:
            results.append(FrameDetections(det_max=0.0, det_label=None))
            continue
        good = [b for b in boxes if b.score >= cfg.min_box_score][: cfg.max_species_boxes]
        names = [b.label for b in good]
        if good and species_model is not None:
            named = species_model.classify([_crop(img, b.xyxy) for b in good])
            names = [
                name if confidence >= settings.gate1.species_min_confidence else box.label
                for box, (name, confidence) in zip(good, named)
            ]
        results.append(FrameDetections(
            det_max=boxes[0].score,
            det_label=boxes[0].label,
            species=list(dict.fromkeys(names)),
        ))
    return results
