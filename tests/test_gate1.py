from time import perf_counter
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from app.core.config import settings
from app.gates.gate1 import gate1_prompts, run_gate1
from app.gates.relevance import relevance_prompts
from app.pipeline import orchestrator
from app.pipeline.context import ModerationContext

LABELS = list(gate1_prompts() | relevance_prompts())   # what the real backbone embeds
SCALE = 100.0  # like the real model (~113): small similarity gaps decide the score


def row(**values) -> np.ndarray:
    """One picture: every prompt 0.1, except labels starting with the given prefixes.

    Keyword names use "__" for "|" and "_SLASH_" for "/", e.g. animal__dogs_SLASH_a dog.
    """
    sims = np.full(len(LABELS), 0.1)
    for prefix, value in values.items():
        prefix = prefix.replace("__", "|").replace("_SLASH_", "/")
        for i, label in enumerate(LABELS):
            if label.startswith(prefix):
                sims[i] = value
    return sims


DOG = {"animal__dogs_SLASH_a dog": 0.3, "photo__": 0.3}
FOOD = {"no_pet__food on a plate": 0.3, "photo__": 0.3}
# one animal idea against one scene idea: share = 1 / (1 + e^(-100 * gap))
EVEN = {"animal__dogs_SLASH_a dog": 0.2, "no_pet__a car": 0.2}                   # 50 -> borderline
SLIGHTLY_MORE = {"animal__dogs_SLASH_a dog": 0.207, "no_pet__a car": 0.2}       # 67


def image(**values):
    return run_gate1(row(**values)[None, :], LABELS, SCALE, is_video=False)


def test_clear_dog_photo_passes():
    result = image(**DOG)

    assert result.outcome == "pass"
    assert result.pet_score >= 75
    assert result.threshold == 75
    assert result.frames[0].best_animal == "dogs/a dog"


def test_plate_of_food_fails():
    result = image(**FOOD)

    assert result.outcome == "fail"
    assert result.pet_score < 50
    assert result.frames[0].best_no_pet == "food on a plate"


def test_two_matching_animals_score_higher_than_one():
    one = image(**SLIGHTLY_MORE)
    two = image(**SLIGHTLY_MORE, **{"animal__cats_SLASH_a kitten": 0.207})

    assert two.pet_score > one.pet_score  # puppy + kitten photos must not lose (spec 2.3)


def test_equal_evidence_is_borderline():
    result = image(**EVEN)

    assert result.pet_score == 50
    assert result.outcome == "borderline"


def test_drawing_uses_threshold_60():
    result = image(**SLIGHTLY_MORE, illustration__=0.4, photo__=0.1)

    assert result.frames[0].illustrated is True
    assert result.threshold == 60
    assert result.outcome == "pass"  # about 67: below 75, above 60


def test_insect_uses_threshold_60():
    result = image(**{"animal__insects_SLASH_a beetle": 0.207, "no_pet__a car": 0.2, "photo__": 0.3})

    assert result.frames[0].insect is True
    assert result.threshold == 60
    assert result.outcome == "pass"


def test_dolphin_in_the_sea_is_wild_context():
    result = image(**{"animal__marine_mammals": 0.3, "wild__": 0.4, "photo__": 0.3})

    assert result.outcome == "pass"
    assert result.wild is True


def test_dog_outdoors_is_never_wild_context():
    assert image(**DOG, wild__=0.4).wild is False


def test_thresholds_come_from_settings(monkeypatch):
    assert image(**SLIGHTLY_MORE, photo__=0.3).outcome == "borderline"  # 67 < 75

    monkeypatch.setattr(settings.gate1.thresholds, "pass_photo", 65)

    assert image(**SLIGHTLY_MORE, photo__=0.3).outcome == "pass"  # 67 >= 65


def video(passing: int, total: int = 10, others=FOOD):
    sims = np.stack([row(**DOG) if i < passing else row(**others) for i in range(total)])
    return run_gate1(sims, LABELS, SCALE, is_video=True)


def test_video_passes_with_30_percent_of_frames():
    result = video(passing=3)

    assert result.outcome == "pass"
    assert result.pass_fraction == 0.3


def test_video_with_too_few_pet_frames_is_borderline():
    assert video(passing=2).outcome == "borderline"


def test_video_without_any_pet_fails():
    assert video(passing=0).outcome == "fail"


# --- in the pipeline ---------------------------------------------------------

class FakeBackbone:
    def __init__(self, values):
        self.values = values
        self.labels = LABELS
        self.scale = SCALE

    def embed_images(self, images):
        return np.zeros((len(images), 4), dtype=np.float32)

    def probabilities(self, embeds):
        return {}

    def similarities(self, embeds):
        return np.stack([row(**self.values)] * len(embeds))


def make_ctx(tmp_path, content_type="image", path=None) -> ModerationContext:
    if path is None:
        path = tmp_path / "x.jpg"
        Image.new("RGB", (64, 64), "orange").save(path, "JPEG")
    return ModerationContext(
        content_id=uuid4(), user_id=uuid4(), content_type=content_type, context="post",
        started_at=perf_counter(), path=path, sha256="a" * 64,
    )


@pytest.fixture
def with_model(monkeypatch):
    monkeypatch.setattr(settings.ml, "backbone", "siglip2_base")

    def use(values):
        monkeypatch.setattr(orchestrator, "get_backbone", lambda: FakeBackbone(values))
    return use


def test_pipeline_pass(tmp_path, with_model):
    with_model(DOG)
    ctx = make_ctx(tmp_path)

    response = orchestrator.run_pipeline(ctx)

    assert response.decision == "APPROVED"
    assert response.flags == ["validation_passed", "gate1_passed", "d1_d4_unchecked"]
    assert response.pet_score >= 75
    assert ctx.gate1.outcome == "pass"


def test_pipeline_fail_is_mod_001_in_live_mode(tmp_path, with_model):
    with_model(FOOD)

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "REJECTED"
    assert response.rejection_code == "MOD_001"
    assert response.hard_reject is False
    assert "no_pet_detected" in response.flags
    assert response.pet_score < 50


def test_pipeline_fail_in_shadow_mode_only_flags(tmp_path, with_model, monkeypatch):
    monkeypatch.setattr(settings.gate1, "enforce", False)
    with_model(FOOD)

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "HUMAN_REVIEW"
    assert response.rejection_code is None
    assert "gate1_would_reject" in response.flags


def test_pipeline_borderline_goes_to_review(tmp_path, with_model):
    with_model(EVEN)

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "HUMAN_REVIEW"
    assert response.flags == ["validation_passed", "gate1_borderline"]


def test_pipeline_wild_context_goes_to_review(tmp_path, with_model):
    with_model({"animal__marine_mammals": 0.3, "wild__": 0.4, "photo__": 0.3})

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "HUMAN_REVIEW"
    assert response.flags == ["validation_passed", "gate1_passed", "wild_context"]


def test_pipeline_video_uses_every_frame(videos, with_model):
    with_model(DOG)
    ctx = make_ctx(None, content_type="video", path=videos["mp4"])

    response = orchestrator.run_pipeline(ctx)

    assert response.flags == ["validation_passed", "gate1_passed", "d1_d4_unchecked"]
    assert len(ctx.gate1.frames) == len(ctx.prepared.frames)
    assert ctx.gate1.pass_fraction == 1.0


def test_pipeline_model_not_loaded_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.ml, "backbone", "siglip2_base")
    monkeypatch.setattr(orchestrator, "get_backbone", lambda: None)

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "HUMAN_REVIEW"
    assert response.flags == ["validation_passed", "gate1_unavailable"]
    assert response.pet_score is None


def test_no_backbone_configured_means_no_gate1(tmp_path):
    response = orchestrator.run_pipeline(make_ctx(tmp_path))  # conftest: backbone None

    assert response.flags == ["validation_passed"]
    assert response.pet_score is None


def test_mod_001_is_not_added_to_the_blocklist(tmp_path, with_model, blocklist):
    with_model(FOOD)

    orchestrator.run_pipeline(make_ctx(tmp_path))

    assert blocklist.count() == 0  # "no pet" is not harmful content


def test_pass_stays_in_review_when_approve_on_pass_is_off(tmp_path, with_model, monkeypatch):
    monkeypatch.setattr(settings.gate1, "approve_on_pass", False)
    with_model(DOG)

    assert orchestrator.run_pipeline(make_ctx(tmp_path)).decision == "HUMAN_REVIEW"


def test_shadow_mode_never_approves(tmp_path, with_model, monkeypatch):
    monkeypatch.setattr(settings.gate1, "enforce", False)
    with_model(DOG)

    assert orchestrator.run_pipeline(make_ctx(tmp_path)).decision == "HUMAN_REVIEW"


def test_approved_answer_is_cached_like_any_decision(client, monkeypatch):
    from tests.test_api import FILE, VALID_USER

    monkeypatch.setattr(settings.ml, "backbone", "siglip2_base")
    monkeypatch.setattr(orchestrator, "get_backbone", lambda: FakeBackbone(DOG))

    def post():
        return client.post("/api/v1/moderate", files=FILE,
                           data={"content_type": "image", "user_id": VALID_USER}).json()

    first = post()
    assert first["decision"] == "APPROVED"
    assert post() == first


def test_gate1_log_line(tmp_path, with_model, caplog):
    import logging

    with_model(DOG)
    with caplog.at_level(logging.INFO, logger="mmp.gate1"):
        orchestrator.run_pipeline(make_ctx(tmp_path))

    line = next(r.getMessage() for r in caplog.records if r.name == "mmp.gate1")
    assert "pet_score=" in line
    assert "best_animal=dogs/a dog (0.300)" in line
    assert "bar=75" in line
    assert "kind=photo" in line
