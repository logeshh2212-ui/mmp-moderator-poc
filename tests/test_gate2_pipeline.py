"""Gate 2 inside the pipeline (D5 + D6 wired in, D1-D4 signals faked) and D6 relevance."""
import logging

import numpy as np
import pytest

from app.core.config import settings
from app.gates.gate2 import FrameSignals
from app.gates.relevance import relevance_prompts, score_relevance
from app.pipeline import orchestrator
from tests.test_gate1 import DOG, EVEN, FOOD, LABELS, SCALE, FakeBackbone, make_ctx, row

PERSONAL = {**DOG, "relevance__personal_moment": 0.3}
ADVERT = {**DOG, "relevance__advert": 0.3}


@pytest.fixture
def with_model(monkeypatch):
    monkeypatch.setattr(settings.ml, "backbone", "siglip2_base")

    def use(values):
        monkeypatch.setattr(orchestrator, "get_backbone", lambda: FakeBackbone(values))
    return use


@pytest.fixture
def detectors_see(monkeypatch):
    """Pretend the (not yet built) D1-D3 detectors saw these signals in every frame."""
    def use(*wanted):
        monkeypatch.setattr(
            orchestrator, "FrameSignals",
            lambda signals=frozenset(), **kw: FrameSignals(signals=set(signals) | set(wanted), **kw),
        )
    return use


# --- D6 relevance on its own ---

def test_relevance_sentences_are_registered_per_kind():
    prompts = relevance_prompts()
    kinds = {label.split("|")[1] for label in prompts}
    assert kinds == set(settings.gate2.relevance.kinds)
    assert all(label.startswith("relevance|") for label in prompts)


def test_most_likely_kind_gives_the_band():
    [result] = score_relevance(row(**{"relevance__meme": 0.3})[None, :], LABELS, SCALE)
    assert result.kind == "meme"
    assert result.confidence > 0.99
    assert result.score == pytest.approx(80, abs=0.5)   # top of the 60-80 band when sure


def test_unsure_relevance_sits_at_the_bottom_of_the_band():
    [result] = score_relevance(row()[None, :], LABELS, SCALE)   # every kind equally likely
    low, _ = settings.gate2.relevance.kinds[result.kind].band
    assert result.confidence == pytest.approx(1 / len(settings.gate2.relevance.kinds))
    assert result.score < low + 5


def test_backbone_without_relevance_sentences_gives_none():
    gate1_only = [label for label in LABELS if not label.startswith("relevance|")]
    assert score_relevance(np.zeros((1, len(gate1_only))), gate1_only, SCALE) is None


# --- in the pipeline ---

def test_pass_gets_all_six_dimensions_and_a_composite(tmp_path, with_model):
    with_model(PERSONAL)
    ctx = make_ctx(tmp_path)

    response = orchestrator.run_pipeline(ctx)

    scores = response.dimension_scores
    assert response.decision == "APPROVED"
    assert (scores.D1_welfare, scores.D2_nsfw, scores.D3_violence, scores.D4_prohibited) == (100, 100, 100, 100)
    assert scores.D6_relevance >= 85
    assert ctx.relevance[0].kind == "personal_moment"
    assert response.composite_score >= settings.gate2.thresholds.approve
    assert "d1_d3_unchecked" in response.flags


def test_quality_uses_the_original_upload_size(tmp_path, with_model):
    with_model(PERSONAL)
    ctx = make_ctx(tmp_path)   # a 64 x 64 test picture: far below the low-resolution bar

    orchestrator.run_pipeline(ctx)

    assert ctx.quality[0].issue in ("screenshot", "blank")
    assert ctx.quality[0].measures.short_side == 64


def test_off_topic_post_is_still_approved_but_scores_lower(tmp_path, with_model):
    with_model(ADVERT)

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "APPROVED"   # decided: D6 never holds a post on its own
    assert response.dimension_scores.D6_relevance <= 40


def test_gate1_failure_skips_gate2(tmp_path, with_model):
    with_model(FOOD)
    ctx = make_ctx(tmp_path)

    response = orchestrator.run_pipeline(ctx)

    assert response.rejection_code == "MOD_001"
    assert response.dimension_scores is None and response.composite_score is None
    assert ctx.gate2 is None


def test_borderline_still_goes_to_a_human_with_scores(tmp_path, with_model):
    with_model(EVEN)

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "HUMAN_REVIEW"
    assert response.dimension_scores is not None


def test_gate2_hard_reject_rejects_the_post(tmp_path, with_model, detectors_see):
    with_model(PERSONAL)
    detectors_see("welfare_fighting")

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "REJECTED"
    assert response.rejection_code == "MOD_003"
    assert response.hard_reject is True
    assert "gate2_rejected" in response.flags


def test_gate2_hard_reject_beats_gate1_borderline(tmp_path, with_model, detectors_see):
    with_model(EVEN)
    detectors_see("nsfw_animal")

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "REJECTED"
    assert response.rejection_code == "MOD_004"


def test_serious_signal_sends_a_pass_to_a_human(tmp_path, with_model, detectors_see):
    with_model(PERSONAL)
    detectors_see("item_drugs")

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "HUMAN_REVIEW"
    assert "gate2_review" in response.flags
    assert response.dimension_scores.D4_prohibited == 20


def test_gate2_shadow_mode_only_flags(tmp_path, with_model, detectors_see, monkeypatch):
    monkeypatch.setattr(settings.gate2, "enforce", False)
    with_model(PERSONAL)
    detectors_see("welfare_fighting")

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "HUMAN_REVIEW"
    assert response.rejection_code is None
    assert "gate2_would_reject" in response.flags


def test_gate2_shadow_mode_never_approves(tmp_path, with_model, monkeypatch):
    monkeypatch.setattr(settings.gate2, "enforce", False)
    with_model(PERSONAL)

    assert orchestrator.run_pipeline(make_ctx(tmp_path)).decision == "HUMAN_REVIEW"


def test_video_gets_quality_and_relevance_for_every_frame(videos, with_model):
    with_model(PERSONAL)
    ctx = make_ctx(None, content_type="video", path=videos["mp4"])

    response = orchestrator.run_pipeline(ctx)

    assert len(ctx.quality) == len(ctx.relevance) == len(ctx.prepared.frames)
    assert response.dimension_scores is not None


def test_gate2_log_line(tmp_path, with_model, caplog):
    with_model(PERSONAL)
    with caplog.at_level(logging.INFO, logger="mmp.gate2"):
        orchestrator.run_pipeline(make_ctx(tmp_path))

    line = next(r.getMessage() for r in caplog.records if r.name == "mmp.gate2")
    assert "verdict=APPROVED" in line
    assert "D1=100" in line
    assert "personal_moment" in line


def test_gate2_time_is_logged(tmp_path, with_model):
    with_model(PERSONAL)
    ctx = make_ctx(tmp_path)

    orchestrator.run_pipeline(ctx)

    assert "gate2" in ctx.timings
