"""D5 image quality (spec 4.5): a 0-100 score from simple picture statistics, no model.

Measured on the detector-size picture (long side 640) that pre-processing already made.
Sharpness and exposure are measured on the pet's box from Gate 1 when there is one, so a
portrait photo with a blurred background, or a pet on a white studio background, is not
marked as blurry or burnt out. D5 never holds or rejects a post on its own (decided); it
only moves the composite and is reported.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import settings

# libjpeg's standard luminance table at quality 50 (natural order, as Pillow returns it).
# A JPEG saved at quality q uses this table scaled by 5000 / q (q < 50) or 200 - 2q
# (q >= 50), so the ratio gives q back.
STANDARD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)


@dataclass
class QualityMeasures:
    contrast: float              # brightness standard deviation 0-255, whole picture
    # The next three are measured on the pet's box when there is one, else the whole picture:
    brightness: float            # mean brightness 0-255
    bright_fraction: float       # share of burnt-out pixels
    sharpness: float             # variance of the Laplacian, scaled to mid-grey brightness
    bar_fraction: float          # share of the picture taken by black bars on both sides
    short_side: int | None       # of the original upload, before any resizing
    jpeg_quality: float | None   # estimated save quality 1-100; None if not a JPEG


@dataclass
class QualityResult:
    score: float     # D5, 0-100
    issue: str       # sharp | slight_blur | heavy_blur | dark | overexposed | screenshot | compressed | blank
    measures: QualityMeasures


def jpeg_quality(path: Path) -> float | None:
    """Estimate the quality a JPEG was saved at, from its quantization table (no decoding)."""
    try:
        with Image.open(path) as img:
            tables = getattr(img, "quantization", None) if img.format == "JPEG" else None
    except OSError:
        return None
    if not tables or 0 not in tables or len(tables[0]) != 64:
        return None
    table = np.asarray(tables[0], dtype=np.float64)
    usable = (table > 1) & (table < 255)  # entries clipped at either end say nothing
    if not usable.any():
        return 100.0 if (table == 1).all() else 1.0
    scale = 100 * table[usable].sum() / STANDARD_LUMA[usable].sum()
    quality = 5000 / scale if scale > 100 else (200 - scale) / 2
    return float(np.clip(quality, 1, 100))


def laplacian_variance(gray: np.ndarray) -> float:
    """How much fine detail there is: low for blurred pictures."""
    if min(gray.shape) < 3:
        return 0.0
    lap = (
        gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1]
        - 4 * gray[1:-1, 1:-1]
    )
    return float(lap.var())


def sharpness(gray: np.ndarray) -> float:
    """Laplacian variance as if the picture had mid-grey brightness: a dark but sharp photo
    has small differences everywhere and would otherwise look blurred (measured: it did)."""
    brightness = max(float(gray.mean()) / 128, 0.05)
    return laplacian_variance(gray) / brightness**2


def bar_fraction(gray: np.ndarray) -> float:
    """Black bars on both opposite edges (a screenshot of a video): their share of the picture."""
    q = settings.gate2.quality

    def run(lines: np.ndarray) -> int:
        dark = (lines.mean(axis=1) < q.bar_level) & (lines.std(axis=1) < q.bar_max_std)
        return int(np.argmin(dark)) if not dark.all() else len(dark)

    rows, cols = gray, gray.T
    top, bottom = run(rows), run(rows[::-1])
    left, right = run(cols), run(cols[::-1])
    return max(2 * min(top, bottom) / gray.shape[0], 2 * min(left, right) / gray.shape[1])


def measure(
    picture: Image.Image,
    box: tuple[float, float, float, float] | None = None,
    short_side: int | None = None,
    jpeg_q: float | None = None,
) -> QualityMeasures:
    q = settings.gate2.quality
    gray = np.asarray(picture.convert("L"), dtype=np.float32)

    region = gray
    if box is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        crop = gray[max(y0, 0):y1, max(x0, 0):x1]
        if min(crop.shape) >= q.min_box_side:  # tiny boxes say little about sharpness
            region = crop

    return QualityMeasures(
        contrast=float(gray.std()),
        brightness=float(region.mean()),
        bright_fraction=float((region >= q.bright_level).mean()),
        sharpness=sharpness(region),
        bar_fraction=bar_fraction(gray),
        short_side=short_side,
        jpeg_quality=jpeg_q,
    )


def in_band(value: float, low: float, high: float, band: tuple[float, float]) -> float:
    """Place value (low..high) inside a spec score band, clipped to the band."""
    t = float(np.clip((value - low) / (high - low), 0, 1))
    return band[0] + t * (band[1] - band[0])


def score_quality(m: QualityMeasures) -> tuple[float, str]:
    """The worst problem found decides the score; no problem = the sharp band."""
    q = settings.gate2.quality
    bands = q.bands
    if m.contrast < q.blank_max_contrast:
        return 0.0, "blank"

    s = q.sharpness
    if m.sharpness < s.heavy_below:
        found = [(in_band(m.sharpness, 0, s.heavy_below, bands.heavy_blur), "heavy_blur")]
    elif m.sharpness < s.slight_below:
        found = [(in_band(m.sharpness, s.heavy_below, s.slight_below, bands.slight_blur), "slight_blur")]
    else:
        found = [(in_band(m.sharpness, s.slight_below, s.full_at, bands.sharp), "sharp")]

    if m.brightness < q.dark_mean_below:
        found.append((in_band(m.brightness, 0, q.dark_mean_below, bands.dark), "dark"))
    if m.bright_fraction > q.bright_fraction_above:
        found.append((in_band(1 - m.bright_fraction, 0, 1 - q.bright_fraction_above, bands.dark), "overexposed"))

    letterboxed = m.bar_fraction >= q.bar_min_fraction
    if m.short_side is not None and (letterboxed or m.short_side < q.lowres_short_side_below):
        content = m.short_side * (1 - m.bar_fraction) if letterboxed else m.short_side
        found.append((in_band(content, 0, q.lowres_short_side_below, bands.screenshot), "screenshot"))

    if m.jpeg_quality is not None and m.jpeg_quality < q.jpeg_quality_below:
        found.append((in_band(m.jpeg_quality, 0, q.jpeg_quality_below, bands.compressed), "compressed"))

    score, issue = min(found)
    return round(score, 1), issue


def frame_quality(
    picture: Image.Image,
    box: tuple[float, float, float, float] | None = None,
    short_side: int | None = None,
    jpeg_q: float | None = None,
) -> QualityResult:
    measures = measure(picture, box, short_side, jpeg_q)
    score, issue = score_quality(measures)
    return QualityResult(score, issue, measures)
