from app.core.config import settings
from app.core.versioning import current_model_version, settings_fingerprint


VALID_USER = "3f2b8c1e-0000-4000-8000-000000000001"


def test_model_version_names_every_part():
    version = current_model_version()

    assert version.startswith(
        f"{settings.model_version}+{settings.gate1.version}+{settings.gate2.version}+"
    )
    assert "+no-model+" in version  # tests run without a real model
    assert "+cfg" in version


def test_model_version_names_model_and_revision(monkeypatch):
    monkeypatch.setattr(settings.ml, "backbone", "siglip2_base")

    assert "+siglip2_base@f775b65+" in current_model_version()


def test_same_settings_same_fingerprint():
    assert settings_fingerprint() == settings_fingerprint()


def test_changing_a_decision_setting_changes_the_fingerprint(monkeypatch):
    before = settings_fingerprint()
    monkeypatch.setattr(settings.gate1, "enforce", not settings.gate1.enforce)

    assert settings_fingerprint() != before


def test_changing_a_prompt_changes_the_fingerprint(monkeypatch):
    before = settings_fingerprint()
    monkeypatch.setattr(settings.ml, "prompts", {**settings.ml.prompts, "cat": "a photo of a kitten"})

    assert settings_fingerprint() != before


def test_runtime_settings_do_not_change_the_fingerprint(monkeypatch):
    before = settings_fingerprint()
    monkeypatch.setattr(settings.redis, "lock_wait_seconds", 99)
    monkeypatch.setattr(settings.ml, "num_threads", 1)
    monkeypatch.setattr(settings.prepare, "ffmpeg_hwaccel", "auto")

    assert settings_fingerprint() == before


def post(client, data: bytes):
    return client.post(
        "/api/v1/moderate",
        files={"file": ("x.jpg", data, "image/jpeg")},
        data={"content_type": "image", "user_id": VALID_USER},
    ).json()


def test_changed_rules_do_not_reuse_old_cached_answers(client, monkeypatch):
    from tests.test_api import make_jpeg

    data = make_jpeg()
    first = post(client, data)
    assert post(client, data)["content_id"] == first["content_id"]  # cached

    new_version = settings.gate1.version + "-changed"
    monkeypatch.setattr(settings.gate1, "version", new_version)
    after_change = post(client, data)

    assert after_change["content_id"] != first["content_id"]  # processed again
    assert after_change["model_version"] != first["model_version"]
    assert new_version in after_change["model_version"]


def test_health_shows_model_version_and_enforce(client):
    body = client.get("/health").json()

    assert body["model_version"] == current_model_version()
    assert body["gate1_enforce"] is settings.gate1.enforce
