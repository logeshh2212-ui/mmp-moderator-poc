"""Step 11: the full pass/reject matrix from the handoff, through the real API."""
import shutil
from pathlib import Path

import pytest

from app.core.config import settings
from scripts.make_fixtures import FIXTURES, build


VALID_USER = "3f2b8c1e-0000-4000-8000-000000000001"


@pytest.fixture(scope="session")
def fx() -> dict[str, Path]:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not on PATH")
    return build(FIXTURES)  # only builds files that are missing


def post(client, path: Path, content_type: str, user_id: str = VALID_USER):
    return client.post(
        "/api/v1/moderate",
        files={"file": (path.name, path.read_bytes())},
        data={"content_type": content_type, "user_id": user_id},
    )


MUST_PASS = [
    ("ok.jpg", "image"),
    ("ok.png", "image"),
    ("ok.webp", "image"),
    ("animated.webp", "image"),
    ("animated.png", "image"),
    ("exif_rotated.jpg", "image"),
    ("ok.mp4", "video"),
    ("ok.mov", "video"),
    ("ok.avi", "video"),
    ("no_audio.mp4", "video"),
    ("exactly_60s.mp4", "video"),
]

MUST_REJECT = [
    ("fake_exe.jpg", "image", "unsupported_format"),
    ("empty.jpg", "image", "empty_file"),
    ("truncated.jpg", "image", "corrupt_file"),
    ("garbage.jpg", "image", "corrupt_file"),
    ("tiny.png", "image", "image_too_small"),
    ("bomb.png", "image", "image_too_large_pixels"),
    ("ok.mp4", "image", "type_mismatch"),
    ("ok.jpg", "video", "type_mismatch"),
    ("long_61s.mp4", "video", "duration_exceeded"),
    ("audio_only.mp4", "video", "no_video_stream"),
    ("corrupt.mp4", "video", "corrupt_file"),
]


@pytest.mark.parametrize("name, content_type", MUST_PASS)
def test_must_pass(client, fx, temp_dir, name, content_type):
    body = post(client, fx[name], content_type).json()

    assert body["decision"] == "HUMAN_REVIEW"
    assert body["flags"] == ["validation_passed"]
    assert list(temp_dir.iterdir()) == []


@pytest.mark.parametrize("name, content_type, flag", MUST_REJECT)
def test_must_reject(client, fx, temp_dir, name, content_type, flag):
    r = post(client, fx[name], content_type)

    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "REJECTED"
    assert body["rejection_code"] == "MOD_008"
    assert body["hard_reject"] is False
    assert body["flags"] == [flag]
    assert body["pet_score"] is None
    assert list(temp_dir.iterdir()) == []


def test_21mb_image_is_too_large(client, fx, tmp_path):
    big = tmp_path / "big.jpg"
    big.write_bytes(fx["ok.jpg"].read_bytes() + b"\0" * settings.limits.image_max_bytes)

    assert post(client, big, "image").json()["flags"] == ["file_too_large"]


def test_iphone_heic_passes(client):
    heic = FIXTURES / "iphone.heic"
    if not heic.exists():
        pytest.skip("copy one real iPhone photo to tests/fixtures/iphone.heic")

    body = post(client, heic, "image").json()

    assert body["decision"] == "HUMAN_REVIEW"
    assert body["flags"] == ["validation_passed"]


def test_missing_user_id_is_422(client, fx):
    r = client.post(
        "/api/v1/moderate",
        files={"file": ("ok.jpg", fx["ok.jpg"].read_bytes())},
        data={"content_type": "image"},
    )
    assert r.status_code == 422


def test_same_file_twice_gives_same_content_id(client, fx):
    first = post(client, fx["ok.jpg"], "image").json()
    second = post(client, fx["ok.jpg"], "image").json()

    assert second["content_id"] == first["content_id"]
    assert second == first
