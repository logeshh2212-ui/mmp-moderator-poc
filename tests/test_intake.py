import asyncio
import hashlib
import io
from uuid import uuid4

import pytest
from fastapi import UploadFile

from app.core.config import settings
from app.validation.errors import ValidationFailure
from app.validation.intake import save_upload


def make_upload(buffer: io.BytesIO) -> UploadFile:
    return UploadFile(file=buffer, filename="upload.bin")


def run_intake(data: bytes, content_type: str = "image"):
    return asyncio.run(save_upload(make_upload(io.BytesIO(data)), content_type, uuid4()))


def test_saves_file_with_size_and_hash(temp_dir):
    data = b"hello world"
    result = run_intake(data)

    assert result.size_bytes == len(data)
    assert result.sha256 == hashlib.sha256(data).hexdigest()
    assert result.path.parent == temp_dir
    assert result.path.read_bytes() == data


def test_empty_file_rejected(temp_dir):
    with pytest.raises(ValidationFailure) as exc:
        run_intake(b"")

    assert exc.value.flag == "empty_file"
    assert list(temp_dir.iterdir()) == []


def test_too_large_rejected_and_cleaned_up(temp_dir, monkeypatch):
    monkeypatch.setattr(settings.limits, "image_max_bytes", 10)

    with pytest.raises(ValidationFailure) as exc:
        run_intake(b"x" * 11)

    assert exc.value.flag == "file_too_large"
    assert list(temp_dir.iterdir()) == []


def test_stops_reading_as_soon_as_limit_is_passed(monkeypatch):
    monkeypatch.setattr(settings.limits, "image_max_bytes", 10)
    monkeypatch.setattr(settings.intake, "chunk_size_bytes", 4)
    buffer = io.BytesIO(b"x" * 1000)

    with pytest.raises(ValidationFailure):
        asyncio.run(save_upload(make_upload(buffer), "image", uuid4()))

    assert buffer.tell() < 1000  # did not read the whole upload


def test_video_uses_video_limit(monkeypatch):
    monkeypatch.setattr(settings.limits, "image_max_bytes", 10)

    result = run_intake(b"x" * 20, content_type="video")

    assert result.size_bytes == 20
