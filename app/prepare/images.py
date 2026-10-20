from pathlib import Path

from PIL import Image, ImageOps

from app.core.config import settings
from app.prepare.media import PreparedFrame, PreparedMedia


RESAMPLE = Image.Resampling.LANCZOS


def to_rgb(img: Image.Image) -> Image.Image:
    """RGB with any transparency flattened onto white (black would look like a dark photo)."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


def fit(img: Image.Image, size: int) -> Image.Image:
    """Resize so the long side is `size`, keeping the aspect ratio (small images grow)."""
    return ImageOps.contain(img, (size, size), RESAMPLE)


def frame_from(img: Image.Image, timestamp: float | None = None) -> PreparedFrame:
    p = settings.prepare
    return PreparedFrame(
        timestamp=timestamp,
        backbone=fit(img, p.backbone_size),
        detector=fit(img, p.detector_size),
    )


def prepare_image(path: Path) -> PreparedMedia:
    """Upright RGB copies at model sizes. Animated images: first frame (decision D-11)."""
    size = settings.prepare.detector_size
    with Image.open(path) as img:
        img.draft("RGB", (size, size))  # JPEG decodes at a reduced scale: fast, low memory
        rgb = to_rgb(ImageOps.exif_transpose(img))
    return PreparedMedia(frames=[frame_from(rgb)])
