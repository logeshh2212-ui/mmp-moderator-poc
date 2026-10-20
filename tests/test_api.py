import hashlib
import io

import pytest
from PIL import Image

from app.cache.idempotency import UploadKey, lock_key, result_key
from app.core.config import settings


def make_jpeg(size=(64, 64)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "orange").save(buffer, "JPEG")
    return buffer.getvalue()


VALID_USER = "3f2b8c1e-0000-4000-8000-000000000001"
FILE = {"file": ("x.jpg", make_jpeg(), "image/jpeg")}

def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_missing_user_id_is_422(client):
    r = client.post("/api/v1/moderate", files=FILE, data={"content_type": "image"})
    assert r.status_code == 422


def test_invalid_content_type_is_422(client):
    r = client.post("/api/v1/moderate", files=FILE,
                    data={"content_type": "audio", "user_id": VALID_USER})
    assert r.status_code == 422


def test_invalid_uuid_is_422(client):
    r = client.post("/api/v1/moderate", files=FILE,
                    data={"content_type": "image", "user_id": "not-a-uuid"})
    assert r.status_code == 422


def test_valid_request_shape(client):
    r = client.post("/api/v1/moderate", files=FILE,
                    data={"content_type": "image", "user_id": VALID_USER})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "HUMAN_REVIEW"
    assert body["pet_score"] is None
    assert body["dimension_scores"] is None
    assert body["species_detected"] == []
    assert body["model_version"] == "validation-0.1"

def test_too_large_image_rejected(client, temp_dir):
    big = b"\0" * (settings.limits.image_max_bytes + 1)  # just over 20 MiB
    r = client.post("/api/v1/moderate",
                    files={"file": ("big.jpg", big, "image/jpeg")},
                    data={"content_type": "image", "user_id": VALID_USER})

    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "REJECTED"
    assert body["rejection_code"] == "MOD_008"
    assert body["flags"] == ["file_too_large"]
    assert body["hard_reject"] is False
    assert list(temp_dir.iterdir()) == []


def test_empty_file_rejected(client, temp_dir):
    r = client.post("/api/v1/moderate",
                    files={"file": ("empty.jpg", b"", "image/jpeg")},
                    data={"content_type": "image", "user_id": VALID_USER})

    assert r.json()["flags"] == ["empty_file"]
    assert list(temp_dir.iterdir()) == []


def test_temp_dir_empty_after_success(client, temp_dir):
    client.post("/api/v1/moderate", files=FILE,
                data={"content_type": "image", "user_id": VALID_USER})

    assert list(temp_dir.iterdir()) == []

def post_image(client, user_id=VALID_USER):
    return client.post("/api/v1/moderate", files=FILE,
                       data={"content_type": "image", "user_id": user_id})


FILE_SHA = hashlib.sha256(FILE["file"][1]).hexdigest()
FILE_KEY = UploadKey(FILE_SHA, VALID_USER, "image", "post")


def test_same_file_twice_returns_same_response(client):
    first = post_image(client).json()
    second = post_image(client).json()

    assert second == first  # same content_id, returned unchanged


def test_same_file_other_user_is_processed_separately(client):
    first = post_image(client).json()
    second = post_image(client, "3f2b8c1e-0000-4000-8000-000000000002").json()

    assert second["content_id"] != first["content_id"]


def test_lock_released_after_request(client, sync_redis):
    post_image(client)


    assert sync_redis.exists(lock_key(FILE_KEY)) == 0
    assert sync_redis.exists(result_key(FILE_KEY)) == 1


def test_locked_upload_returns_409(client, sync_redis, temp_dir, monkeypatch):
    monkeypatch.setattr(settings.redis, "lock_poll_seconds", 0.01)
    monkeypatch.setattr(settings.redis, "lock_wait_seconds", 0.05)
    sync_redis.set(lock_key(FILE_KEY), "1", ex=60)

    r = post_image(client)

    assert r.status_code == 409
    assert list(temp_dir.iterdir()) == []


def test_too_large_rejection_is_not_cached(client, sync_redis):
    big = b"\0" * (settings.limits.image_max_bytes + 1)
    client.post("/api/v1/moderate",
                files={"file": ("big.jpg", big, "image/jpeg")},
                data={"content_type": "image", "user_id": VALID_USER})

    assert sync_redis.keys("mod:*") == []


def test_errors_are_not_cached_and_lock_is_released(client, sync_redis, monkeypatch):
    def crash(path, content_type):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.api.routes.run_checks", crash)

    with pytest.raises(RuntimeError):
        post_image(client)

    assert sync_redis.exists(result_key(FILE_KEY)) == 0
    assert sync_redis.exists(lock_key(FILE_KEY)) == 0

def make_jpeg(size=(64, 64)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "orange").save(buffer, "JPEG")
    return buffer.getvalue()



def test_exe_renamed_to_jpg_rejected(client, temp_dir):
    exe = b"MZ\x90\x00" + b"\x00" * 60
    r = client.post("/api/v1/moderate",
                    files={"file": ("fake.jpg", exe, "image/jpeg")},
                    data={"content_type": "image", "user_id": VALID_USER})

    body = r.json()
    assert body["decision"] == "REJECTED"
    assert body["rejection_code"] == "MOD_008"
    assert body["flags"] == ["unsupported_format"]
    assert list(temp_dir.iterdir()) == []


def test_mp4_sent_as_image_is_type_mismatch(client):
    mp4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 40
    r = client.post("/api/v1/moderate",
                    files={"file": ("clip.jpg", mp4, "image/jpeg")},
                    data={"content_type": "image", "user_id": VALID_USER})

    assert r.json()["flags"] == ["type_mismatch"]


def test_jpeg_sent_as_video_is_type_mismatch(client):
    r = client.post("/api/v1/moderate", files=FILE,
                    data={"content_type": "video", "user_id": VALID_USER})

    assert r.json()["flags"] == ["type_mismatch"]


def test_format_rejection_is_cached(client, sync_redis):
    exe = b"MZ\x90\x00" + b"\x00" * 60
    first = client.post("/api/v1/moderate",
                        files={"file": ("fake.jpg", exe, "image/jpeg")},
                        data={"content_type": "image", "user_id": VALID_USER}).json()
    second = client.post("/api/v1/moderate",
                         files={"file": ("fake.jpg", exe, "image/jpeg")},
                         data={"content_type": "image", "user_id": VALID_USER}).json()

    assert second == first  # a format rejection is a final decision, so it is stored

def test_same_file_as_video_then_image_is_not_a_duplicate(client):
    as_video = client.post("/api/v1/moderate", files=FILE,
                           data={"content_type": "video", "user_id": VALID_USER}).json()
    as_image = post_image(client).json()

    assert as_video["flags"] == ["type_mismatch"]
    assert as_image["decision"] == "HUMAN_REVIEW"
    assert as_image["content_id"] != as_video["content_id"]


def test_same_file_other_context_is_not_a_duplicate(client):
    as_post = post_image(client).json()
    as_breeding = client.post("/api/v1/moderate", files=FILE,
                              data={"content_type": "image", "user_id": VALID_USER,
                                    "context": "breeding"}).json()

    assert as_breeding["content_id"] != as_post["content_id"]

def post_bytes(client, data: bytes, name: str = "img.jpg"):
    return client.post("/api/v1/moderate",
                       files={"file": (name, data, "image/jpeg")},
                       data={"content_type": "image", "user_id": VALID_USER})


def test_truncated_jpeg_is_corrupt_file(client, temp_dir):
    data = make_jpeg((256, 256))
    body = post_bytes(client, data[: len(data) // 2]).json()

    assert body["decision"] == "REJECTED"
    assert body["flags"] == ["corrupt_file"]
    assert list(temp_dir.iterdir()) == []


def test_tiny_image_is_too_small(client):
    assert post_bytes(client, make_jpeg((16, 16))).json()["flags"] == ["image_too_small"]


def test_huge_png_is_too_large_pixels(client):
    from tests.test_image_check import png_header_only

    body = post_bytes(client, png_header_only(50_000, 50_000), "bomb.png").json()

    assert body["flags"] == ["image_too_large_pixels"]
def post_video(client, path):
    return client.post("/api/v1/moderate",
                       files={"file": (path.name, path.read_bytes(), "video/mp4")},
                       data={"content_type": "video", "user_id": VALID_USER})


def test_valid_video_goes_to_review(client, videos, temp_dir):
    body = post_video(client, videos["mp4"]).json()

    assert body["decision"] == "HUMAN_REVIEW"
    assert list(temp_dir.iterdir()) == []


@pytest.mark.parametrize("name, flag", [
    ("61s", "duration_exceeded"),
    ("audio_only", "no_video_stream"),
    ("corrupt", "corrupt_file"),
])
def test_bad_videos_rejected(client, videos, name, flag):
    body = post_video(client, videos[name]).json()

    assert body["decision"] == "REJECTED"
    assert body["rejection_code"] == "MOD_008"
    assert body["flags"] == [flag]


def test_timeout_is_503_and_not_cached(client, videos, sync_redis, monkeypatch):
    monkeypatch.setattr(settings.timeouts, "ffprobe_seconds", 0.001)

    r = post_video(client, videos["mp4"])

    assert r.status_code == 503
    assert sync_redis.keys("mod:*") == []  # no stored answer, no lock left behind


