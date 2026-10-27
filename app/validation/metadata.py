import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from app.core.config import settings


# EXIF tag numbers (see PIL.ExifTags).
MAKE = 0x010F
MODEL = 0x0110
SOFTWARE = 0x0131
DATETIME = 0x0132
EXIF_IFD = 0x8769
GPS_IFD = 0x8825
DATETIME_ORIGINAL = 0x9003

# PNG text chunks that AI image tools write (Stable Diffusion, ComfyUI, InvokeAI).
AI_PNG_TEXT_KEYS = {"parameters", "prompt", "workflow", "invokeai_metadata", "sd-metadata"}

# IPTC "digital source type" values found in XMP. Case matters: they do not overlap.
XMP_AI_MARKERS = {
    "trainedAlgorithmicMedia": "xmp:trained_algorithmic_media",
    "compositeWithTrainedAlgorithmicMedia": "xmp:composite_with_ai",
}

# ffprobe tags that mean the video has a location. We never keep the coordinates.
VIDEO_GPS_TAGS = ("location", "location-eng", "com.apple.quicktime.location.iso6709")

MAX_TEXT_LENGTH = 200


@dataclass
class MediaMetadata:
    camera_make: str | None = None
    camera_model: str | None = None
    software: str | None = None
    taken_at: str | None = None
    has_gps: bool = False  # only whether it exists; coordinates are never stored
    text_keys: list[str] = field(default_factory=list)
    ai_hints: list[str] = field(default_factory=list)


def clean(value) -> str | None:
    """Turn an EXIF/tag value into short, safe text, or None."""
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).replace("\x00", "").strip()
    return text[:MAX_TEXT_LENGTH] or None


def generator_hints(*values: str | None) -> list[str]:
    """Generator names as whole words only: "imagen" must not match the camera tag
    "aux:ImageNumber" that Photoshop writes into XMP."""
    hints: list[str] = []
    for value in values:
        if not value:
            continue
        for word in settings.metadata.ai_generator_keywords:
            hint = f"generator:{word}"
            if re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", value, re.IGNORECASE) and hint not in hints:
                hints.append(hint)
    return hints


def xmp_text(info: dict) -> str:
    raw = info.get("xmp") or info.get("XML:com.adobe.xmp") or ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return raw


def read_image_metadata(path: Path) -> MediaMetadata:
    """Read EXIF, XMP and PNG text. Never raises: bad metadata just means less information."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            exif_ifd = exif.get_ifd(EXIF_IFD)
            gps_ifd = exif.get_ifd(GPS_IFD)
            xmp = xmp_text(img.info)
            text = dict(img.text) if img.format == "PNG" else {}

        software = clean(exif.get(SOFTWARE)) or clean(text.get("Software"))

        hints = [hint for marker, hint in XMP_AI_MARKERS.items() if marker in xmp]
        hints += [f"png_text:{key}" for key in sorted(text) if key.lower() in AI_PNG_TEXT_KEYS]
        hints += generator_hints(software, xmp)

        return MediaMetadata(
            camera_make=clean(exif.get(MAKE)),
            camera_model=clean(exif.get(MODEL)),
            software=software,
            taken_at=clean(exif_ifd.get(DATETIME_ORIGINAL)) or clean(exif.get(DATETIME)),
            has_gps=bool(gps_ifd),
            text_keys=sorted(text),
            ai_hints=hints,
        )
    except Exception:
        return MediaMetadata()


def read_video_metadata(tags: dict) -> MediaMetadata:
    """Read camera, software and date from ffprobe format tags. Never raises."""
    try:
        tags = {str(key).lower(): value for key, value in tags.items()}
        software = clean(tags.get("com.apple.quicktime.software")) or clean(tags.get("encoder"))

        return MediaMetadata(
            camera_make=clean(tags.get("com.apple.quicktime.make") or tags.get("com.android.manufacturer")),
            camera_model=clean(tags.get("com.apple.quicktime.model") or tags.get("com.android.model")),
            software=software,
            taken_at=clean(tags.get("com.apple.quicktime.creationdate") or tags.get("creation_time")),
            has_gps=any(key in tags for key in VIDEO_GPS_TAGS),
            text_keys=sorted(tags),
            ai_hints=generator_hints(software, clean(tags.get("comment"))),
        )
    except Exception:
        return MediaMetadata()
