import io
import struct
import zlib

import pytest
from PIL import Image

from app.core.config import settings
from app.validation.errors import ValidationFailure
from app.validation.image_check import check_image


def save(tmp_path, img: Image.Image, fmt: str, name: str = "img", **kwargs):
    path = tmp_path / f"{name}.{fmt.lower()}"
    img.save(path, fmt, **kwargs)
    return path


def noisy(size=(256, 256)) -> Image.Image:
    """A detailed image, so a cut-off copy really is missing pixel data."""
    return Image.effect_noise(size, 64).convert("RGB")


def png_header_only(width: int, height: int) -> bytes:
    """A tiny PNG that claims to be width x height. No pixel data at all."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_valid_images_pass(tmp_path, fmt):
    info = check_image(save(tmp_path, noisy((120, 80)), fmt))

    assert (info.width, info.height) == (120, 80)
    assert info.format == fmt
    assert info.animated is False
    assert info.frame_count == 1


def test_animated_webp_passes(tmp_path):
    frames = [Image.new("RGB", (64, 64), color) for color in ("red", "green", "blue")]
    path = save(tmp_path, frames[0], "WEBP", save_all=True, append_images=frames[1:])

    info = check_image(path)

    assert info.animated is True
    assert info.frame_count == 3


def test_animated_png_passes(tmp_path):
    frames = [Image.new("RGB", (64, 64), color) for color in ("red", "green")]
    path = save(tmp_path, frames[0], "PNG", save_all=True, append_images=frames[1:])

    info = check_image(path)

    assert info.animated is True
    assert info.frame_count == 2


def test_heic_passes(tmp_path):
    path = tmp_path / "img.heic"
    try:
        noisy((128, 96)).save(path, "HEIF")
    except Exception as exc:  # encoder not available in this build
        pytest.skip(f"cannot write HEIC here: {exc}")

    info = check_image(path)

    assert (info.width, info.height) == (128, 96)
    assert info.format == "HEIF"


def test_exif_rotation_swaps_width_and_height(tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees
    path = save(tmp_path, noisy((100, 50)), "JPEG", exif=exif)

    info = check_image(path)

    assert info.orientation == 6
    assert (info.width, info.height) == (50, 100)


def test_exactly_min_side_passes(tmp_path):
    side = settings.limits.min_image_side_pixels
    info = check_image(save(tmp_path, Image.new("RGB", (side, side)), "PNG"))

    assert info.width == side


@pytest.mark.parametrize("size", [(16, 16), (31, 100), (100, 31)])
def test_too_small_rejected(tmp_path, size):
    with pytest.raises(ValidationFailure) as exc:
        check_image(save(tmp_path, Image.new("RGB", size), "PNG"))
    assert exc.value.flag == "image_too_small"


def test_truncated_jpeg_rejected(tmp_path):
    buffer = io.BytesIO()
    noisy().save(buffer, "JPEG")
    data = buffer.getvalue()
    path = tmp_path / "truncated.jpg"
    path.write_bytes(data[: len(data) // 2])

    with pytest.raises(ValidationFailure) as exc:
        check_image(path)
    assert exc.value.flag == "corrupt_file"


def test_garbage_after_jpeg_magic_rejected(tmp_path):
    path = tmp_path / "garbage.jpg"
    path.write_bytes(b"\xff\xd8\xff" + b"not really a jpeg" * 50)

    with pytest.raises(ValidationFailure) as exc:
        check_image(path)
    assert exc.value.flag == "corrupt_file"


def test_50000_by_50000_png_rejected(tmp_path):
    path = tmp_path / "bomb.png"
    path.write_bytes(png_header_only(50_000, 50_000))

    with pytest.raises(ValidationFailure) as exc:
        check_image(path)
    assert exc.value.flag == "image_too_large_pixels"


def test_just_over_pixel_limit_rejected(tmp_path, monkeypatch):
    # Between the limit and 2x the limit Pillow only warns; we must still reject.
    monkeypatch.setattr(settings.limits, "image_max_pixels", 1000)

    with pytest.raises(ValidationFailure) as exc:
        check_image(save(tmp_path, Image.new("RGB", (40, 40)), "PNG"))  # 1600 px
    assert exc.value.flag == "image_too_large_pixels"
