import pytest

from app.core.config import settings
from app.validation.errors import ValidationFailure
from app.validation.formats import check_format, detect_format


def pad(data: bytes) -> bytes:
    return data + b"\x00" * (32 - len(data))


def ftyp(brand: bytes) -> bytes:
    return pad(b"\x00\x00\x00\x18ftyp" + brand)


JPEG = pad(b"\xff\xd8\xff\xe0")
PNG = pad(b"\x89PNG\r\n\x1a\n")
WEBP = pad(b"RIFF\x00\x00\x00\x00WEBP")
AVI = pad(b"RIFF\x00\x00\x00\x00AVI ")


@pytest.mark.parametrize("header, expected", [
    (JPEG, "jpeg"),
    (PNG, "png"),
    (WEBP, "webp"),
    (AVI, "avi"),
    (ftyp(b"heic"), "heic"),
    (ftyp(b"heix"), "heic"),
    (ftyp(b"mif1"), "heic"),
    (ftyp(b"msf1"), "heic"),
    (ftyp(b"isom"), "mp4"),
    (ftyp(b"iso2"), "mp4"),
    (ftyp(b"mp41"), "mp4"),
    (ftyp(b"mp42"), "mp4"),
    (ftyp(b"avc1"), "mp4"),
    (ftyp(b"M4V "), "mp4"),
    (ftyp(b"qt  "), "mov"),
])
def test_detects_known_formats(header, expected):
    assert detect_format(header) == expected


@pytest.mark.parametrize("header", [
    pad(b"MZ\x90\x00"),               # Windows .exe
    pad(b"GIF89a"),                   # GIF is not allowed
    pad(b"%PDF-1.7"),                 # PDF
    pad(b"RIFF\x00\x00\x00\x00WAVE"), # WAV audio
    ftyp(b"avif"),                    # AVIF is not in the list
    b"\xff",                          # too short
    b"",                              # empty
])
def test_unknown_formats_return_none(header):
    assert detect_format(header) is None


def write(tmp_path, data: bytes):
    path = tmp_path / "upload.bin"
    path.write_bytes(data)
    return path


def test_check_format_accepts_matching_type(tmp_path):
    assert check_format(write(tmp_path, JPEG), "image") == "jpeg"
    assert check_format(write(tmp_path, ftyp(b"qt  ")), "video") == "mov"


def test_check_format_unsupported(tmp_path):
    with pytest.raises(ValidationFailure) as exc:
        check_format(write(tmp_path, pad(b"MZ")), "image")
    assert exc.value.flag == "unsupported_format"


def test_check_format_image_declared_as_video(tmp_path):
    with pytest.raises(ValidationFailure) as exc:
        check_format(write(tmp_path, PNG), "video")
    assert exc.value.flag == "type_mismatch"


def test_check_format_video_declared_as_image(tmp_path):
    with pytest.raises(ValidationFailure) as exc:
        check_format(write(tmp_path, ftyp(b"mp42")), "image")
    assert exc.value.flag == "type_mismatch"


def test_format_removed_from_settings_is_unsupported(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.allowed_formats, "image", ["jpeg", "png", "webp"])

    with pytest.raises(ValidationFailure) as exc:
        check_format(write(tmp_path, ftyp(b"heic")), "image")
    assert exc.value.flag == "unsupported_format"
