from app.core.config import settings
import hashlib

import pytest

from app.cache.idempotency import lock_key, result_key

VALID_USER = "3f2b8c1e-0000-4000-8000-000000000001"
FILE = {"file": ("x.jpg", b"fake", "image/jpeg")}


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

    assert sync_redis.exists(lock_key(FILE_SHA, VALID_USER)) == 0
    assert sync_redis.exists(result_key(FILE_SHA, VALID_USER)) == 1


def test_locked_upload_returns_409(client, sync_redis, temp_dir, monkeypatch):
    monkeypatch.setattr(settings.redis, "lock_poll_seconds", 0.01)
    monkeypatch.setattr(settings.redis, "lock_wait_seconds", 0.05)
    sync_redis.set(lock_key(FILE_SHA, VALID_USER), "1", ex=60)

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

    assert sync_redis.exists(result_key(FILE_SHA, VALID_USER)) == 0
    assert sync_redis.exists(lock_key(FILE_SHA, VALID_USER)) == 0


