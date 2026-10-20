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
