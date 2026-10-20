import logging
from time import perf_counter
from uuid import uuid4

from PIL import Image

from app.pipeline import orchestrator
from app.pipeline.context import ModerationContext
from app.validation.csam import CsamResult, PlaceholderCsamChecker, get_csam_checker


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
