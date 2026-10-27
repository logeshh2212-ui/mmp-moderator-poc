import hashlib
import json
import logging
from time import perf_counter
from uuid import uuid4

import pytest
from PIL import Image

from app.core.config import SafetySettings, settings
from app.pipeline import orchestrator
from app.pipeline.context import ModerationContext
from app.validation.csam import (
    CsamCheckError, CsamResult, HashListCsamChecker, PlaceholderCsamChecker, get_csam_checker,
)


class AlwaysMatches:
    """Fake checker. Pretends to find a match so we can test the pipeline path.
    No real material is ever used (handoff rule 10)."""

    def check(self, path, sha256):
        return CsamResult(checked=True, match=True, provider="fake")


def make_ctx(tmp_path) -> ModerationContext:
    path = tmp_path / "x.jpg"
    Image.new("RGB", (64, 64), "orange").save(path, "JPEG")
    return ModerationContext(
        content_id=uuid4(),
        user_id=uuid4(),
        content_type="image",
        context="post",
        started_at=perf_counter(),
        path=path,
        sha256="a" * 64,
    )


def test_placeholder_never_matches_and_says_not_checked(tmp_path):
    result = PlaceholderCsamChecker().check(tmp_path / "x.jpg", "a" * 64)

    assert result == CsamResult(checked=False, match=False, provider="placeholder")


def test_placeholder_warns_when_created(caplog):
    with caplog.at_level(logging.WARNING, logger="mmp.safety"):
        PlaceholderCsamChecker()

    assert "PLACEHOLDER" in caplog.text


def test_checker_is_created_once():
    assert get_csam_checker() is get_csam_checker()


def test_valid_file_is_checked_and_passes(tmp_path):
    ctx = make_ctx(tmp_path)

    response = orchestrator.run_pipeline(ctx)

    assert ctx.csam is not None
    assert ctx.csam.match is False
    assert response.flags == ["validation_passed"]


def test_match_is_quarantined_and_stops_the_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "get_csam_checker", lambda: AlwaysMatches())
    ctx = make_ctx(tmp_path)

    response = orchestrator.run_pipeline(ctx)

    assert response.decision == "REJECTED"
    assert response.rejection_code == "MOD_004"
    assert response.hard_reject is True
    assert response.flags == ["quarantined"]
    assert "validation_passed" not in response.flags


def test_invalid_file_never_reaches_csam_check(tmp_path, monkeypatch):
    calls = []

    class Spy:
        def check(self, path, sha256):
            calls.append(path)
            return CsamResult(checked=False, match=False, provider="spy")

    monkeypatch.setattr(orchestrator, "get_csam_checker", lambda: Spy())
    ctx = make_ctx(tmp_path)
    ctx.path.write_bytes(b"not an image")

    orchestrator.run_pipeline(ctx)

    assert calls == []


# --- quarantine: evidence is kept, not deleted ---

def test_match_is_moved_to_quarantine_with_a_report_record(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "get_csam_checker", lambda: AlwaysMatches())
    ctx = make_ctx(tmp_path)
    upload = ctx.path

    orchestrator.run_pipeline(ctx)

    folder = settings.safety.quarantine_dir
    assert not upload.exists()                                  # no longer in the temp folder
    assert (folder / f"{ctx.content_id}.jpg").exists()
    record = json.loads((folder / f"{ctx.content_id}.json").read_text())
    assert record["sha256"] == "a" * 64 and record["provider"] == "fake"
    assert "quarantined_at" in record


# --- hash list provider (exact SHA-256 / MD5) ---

def hash_list(tmp_path, *hashes):
    path = tmp_path / "hashes.txt"
    path.write_text("# test list: hashes of harmless generated files only\n" + "\n".join(hashes))
    return path


def test_hash_list_matches_by_sha256(tmp_path):
    checker = HashListCsamChecker(hash_list(tmp_path, "B" * 64))

    assert checker.check(tmp_path / "x.jpg", "b" * 64) == CsamResult(checked=True, match=True, provider="hash_list")
    assert checker.check(tmp_path / "x.jpg", "c" * 64).match is False


def test_hash_list_matches_by_md5_of_the_file(tmp_path):
    upload = tmp_path / "x.jpg"
    upload.write_bytes(b"harmless test bytes")
    checker = HashListCsamChecker(hash_list(tmp_path, hashlib.md5(b"harmless test bytes").hexdigest()))

    assert checker.check(upload, "c" * 64).match is True


def test_unreadable_hash_list_stops_startup(tmp_path):
    with pytest.raises(CsamCheckError):
        HashListCsamChecker(tmp_path / "missing.txt")


def test_placeholder_is_refused_when_a_real_check_is_required():
    data = settings.safety.model_dump()
    data["require_real_check"] = True
    with pytest.raises(ValueError, match="placeholder"):
        SafetySettings.model_validate(data)


def test_hash_list_provider_needs_a_file():
    data = settings.safety.model_dump()
    data["csam_provider"] = "hash_list"
    with pytest.raises(ValueError, match="hash_list_file"):
        SafetySettings.model_validate(data)


# --- the check cannot run: no decision, retry ---

def test_failed_check_answers_503_and_is_not_cached(client, monkeypatch):
    from tests.test_api import make_jpeg

    class Broken:
        def check(self, path, sha256):
            raise CsamCheckError("provider timed out")

    monkeypatch.setattr(orchestrator, "get_csam_checker", lambda: Broken())
    data = make_jpeg()
    post = lambda: client.post("/api/v1/moderate", files={"file": ("x.jpg", data, "image/jpeg")},  # noqa: E731
                               data={"content_type": "image", "user_id": str(uuid4())})

    assert post().status_code == 503
    monkeypatch.setattr(orchestrator, "get_csam_checker", lambda: PlaceholderCsamChecker())
    assert post().status_code == 200   # a retry is processed: the 503 was not stored
