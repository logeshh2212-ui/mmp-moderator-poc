from app.core.config import settings

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
