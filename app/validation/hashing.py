from io import BytesIO
from pathlib import Path

import imagehash
from PIL import Image


# pHash itself only looks at a 32x32 version, so shrink first to save time and memory.
THUMBNAIL_SIZE = (512, 512)

# Same mapping as PIL.ImageOps.exif_transpose, applied to the small copy only.
ORIENTATION_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def _phash(img: Image.Image, orientation: int = 1) -> str:
    img.thumbnail(THUMBNAIL_SIZE)  # first frame if animated
    small = img.convert("RGB")
    method = ORIENTATION_TRANSPOSE.get(orientation)
    if method is not None:
        small = small.transpose(method)
    return str(imagehash.phash(small))


def phash_image(path: Path, orientation: int = 1) -> str | None:
    """64-bit perceptual hash as 16 hex characters. None if it cannot be computed."""
    try:
        with Image.open(path) as img:
            return _phash(img, orientation)
    except Exception:
        return None


def phash_png_bytes(png: bytes) -> str | None:
    """Perceptual hash of a decoded video frame (from Step 7)."""
    try:
        with Image.open(BytesIO(png)) as img:
            return _phash(img)
    except Exception:
        return None


def hamming(a: str, b: str) -> int:
    """How many of the 64 bits differ. Around 8 or less means 'same picture'."""
    return int(imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b))
