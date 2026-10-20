import warnings
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener

from app.core.config import settings
from app.validation.errors import ValidationFailure


# Runs once, when this module is first imported at startup.
register_heif_opener()

# We check the pixel count ourselves below, so Pillow's warning is just noise.
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

EXIF_ORIENTATION = 0x0112
# Orientations 5-8 rotate the picture by 90 degrees, so width and height swap.
ROTATED_ORIENTATIONS = {5, 6, 7, 8}


@dataclass
class ImageInfo:
    width: int          # as displayed, after EXIF rotation
    height: int
    format: str         # Pillow's name: JPEG, PNG, WEBP, HEIF
    animated: bool
    frame_count: int
    orientation: int    # EXIF orientation 1-8 (1 = normal)


def check_image(path: Path) -> ImageInfo:
    """Open, verify and fully decode the image. Raise ValidationFailure if it is bad."""
    max_pixels = settings.limits.image_max_pixels
    Image.MAX_IMAGE_PIXELS = max_pixels

    try:
        # Pass 1: open reads only the header, so huge sizes are caught before decoding.
        with Image.open(path) as img:
            width, height = img.size
            if width * height > max_pixels:
                raise ValidationFailure("image_too_large_pixels")
            img.verify()

        # Pass 2: verify() leaves the image unusable, so reopen and decode every pixel.
        with Image.open(path) as img:
            img.load()
            file_format = img.format or "unknown"
            animated = bool(getattr(img, "is_animated", False))
            frame_count = int(getattr(img, "n_frames", 1))
            orientation = int(img.getexif().get(EXIF_ORIENTATION, 1))

    except ValidationFailure:
        raise
    except Image.DecompressionBombError:
        raise ValidationFailure("image_too_large_pixels")
    except Exception:
        # Any decode problem: truncated, broken, or not really an image.
        raise ValidationFailure("corrupt_file")

    if orientation in ROTATED_ORIENTATIONS:
        width, height = height, width

    min_side = settings.limits.min_image_side_pixels
    if width < min_side or height < min_side:
        raise ValidationFailure("image_too_small")

    return ImageInfo(
        width=width,
        height=height,
        format=file_format,
        animated=animated,
        frame_count=frame_count,
        orientation=orientation,
    )
