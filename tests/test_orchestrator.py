import logging
from time import perf_counter
from uuid import uuid4

import pytest
from PIL import Image

from app.pipeline import orchestrator
from app.pipeline.context import ModerationContext
from app.validation.errors import ProcessingTimeout


def make_ctx(path, content_type="image") -> ModerationContext:
    return ModerationContext(
        content_id=uuid4(),
        user_id=uuid4(),
        content_type=content_type,
        context="post",
        started_at=perf_counter(),
        path=path,
    )


def jpeg(tmp_path, size=(64, 64)):
    path = tmp_path / "x.jpg"
    Image.new("RGB", size, "orange").save(path, "JPEG")
    return path


def test_valid_file_goes_to_review_with_validation_passed(tmp_path):
    ctx = make_ctx(jpeg(tmp_path))

    response = orchestrator.run_pipeline(ctx)

    assert response.decision == "HUMAN_REVIEW"
    assert response.flags == ["validation_passed"]
    assert response.rejection_code is None
    assert response.hard_reject is False
    assert response.pet_score is None
    assert response.dimension_scores is None
    assert response.ai_generated is None
    assert response.content_id == ctx.content_id
    assert ctx.validation is not None
    assert ctx.validation.file_format == "jpeg"


def test_bad_file_is_rejected_with_mod_008(tmp_path):
    ctx = make_ctx(jpeg(tmp_path, size=(16, 16)))

    response = orchestrator.run_pipeline(ctx)

    assert response.decision == "REJECTED"
    assert response.rejection_code == "MOD_008"
    assert response.flags == ["image_too_small"]
    assert response.hard_reject is False
    assert response.pet_score is None
    assert ctx.validation is None


def test_timeout_is_not_turned_into_a_decision(tmp_path, monkeypatch):
    def slow(path, content_type):
        raise ProcessingTimeout("ffprobe timed out")

    monkeypatch.setattr(orchestrator, "run_checks", slow)

    with pytest.raises(ProcessingTimeout):
        orchestrator.run_pipeline(make_ctx(jpeg(tmp_path)))


def test_one_log_line_per_request(tmp_path, caplog):
    ctx = make_ctx(jpeg(tmp_path, size=(16, 16)))

    with caplog.at_level(logging.INFO, logger="mmp.moderation"):
        orchestrator.run_pipeline(ctx)

    lines = [r.getMessage() for r in caplog.records if r.name == "mmp.moderation"]
    assert len(lines) == 1
    assert str(ctx.content_id) in lines[0]
    assert "decision=REJECTED" in lines[0]
    assert "reason=MOD_008:image_too_small" in lines[0]
    assert "ms=" in lines[0]


def test_processing_ms_is_measured(tmp_path):
    ctx = make_ctx(jpeg(tmp_path))
    ctx.started_at -= 0.25  # pretend the request started 250 ms ago

    assert orchestrator.run_pipeline(ctx).processing_ms >= 250
