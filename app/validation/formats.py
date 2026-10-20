from pathlib import Path

from app.core.config import settings
from app.validation.errors import ValidationFailure


# All signatures below fit in the first 32 bytes of a file.
HEADER_BYTES = 32

HEIC_BRANDS = {b"heic", b"heix", b"mif1", b"msf1"}
MP4_BRANDS = {b"isom", b"iso2", b"mp41", b"mp42", b"avc1", b"M4V "}
MOV_BRANDS = {b"qt  "}


def detect_format(header: bytes) -> str | None:
    """Return the real format from the file's first bytes, or None if unknown."""
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header[:4] == b"RIFF":
        if header[8:12] == b"WEBP":
            return "webp"
        if header[8:12] == b"AVI ":
            return "avi"
    if header[4:8] == b"ftyp":
        brand = header[8:12]
        if brand in HEIC_BRANDS:
            return "heic"
        if brand in MP4_BRANDS:
            return "mp4"
        if brand in MOV_BRANDS:
            return "mov"
    return None


def check_format(path: Path, content_type: str) -> str:
    """Detect the real format and make sure it matches the declared content_type."""
    with path.open("rb") as file:
        header = file.read(HEADER_BYTES)

    file_format = detect_format(header)
    allowed = settings.allowed_formats

    if file_format in allowed.image:
        real_type = "image"
    elif file_format in allowed.video:
        real_type = "video"
    else:
        raise ValidationFailure("unsupported_format")

    if real_type != content_type:
        raise ValidationFailure("type_mismatch")

    return file_format
