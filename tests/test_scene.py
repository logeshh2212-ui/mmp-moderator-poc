"""Step 2e: D1-D3 scenes from SigLIP sentences, D2 human nudity from the NSFW model."""
import numpy as np
import pytest

from app.core.config import Gate2Settings, settings
from app.gates.nsfw import nsfw_signals
from app.gates.scene import scene_prompts, score_scene
from app.pipeline import orchestrator
from tests.test_detection import SIGLIP_SURE, box, models  # noqa: F401
from tests.test_gate1 import BIAS, LABELS, SCALE, make_ctx, row
from tests.test_gate2_pipeline import PERSONAL, with_model  # noqa: F401


def scene(**values):
    [result] = score_scene(row(**values)[None, :], LABELS, SCALE, BIAS)
    return result


# --- SigLIP scenes ---

def test_every_scene_sentence_is_registered():
    groups = {label.split("|")[0] for label in scene_prompts()}
    assert groups == {"scene_safe", "scene_harm", "scene_context"}


def test_ordinary_picture_has_no_scene_signals():
    result = scene()
    assert result.signals == set() and result.contexts == set()


def test_matching_harm_sentence_gives_every_signal_of_its_idea():
    result = scene(scene_harm__abuse=0.3)
    assert result.signals == {"welfare_abuse", "violence_human_on_animal"}
    assert result.harm["abuse"] > 0.99


def test_harm_sentence_that_a_safe_sentence_beats_does_not_count():
    # e.g. puppies play fighting: "dogs fighting" fits, "puppies play fighting" fits better
    assert scene(scene_harm__fighting=0.3, scene_safe__play=0.35).signals == set()


def test_harm_below_min_probability_does_not_count():
    # yes-probability = 1 / (1 + e^-(0.15 * 100 - 16.8)) = 0.14 < 0.2
    assert scene(scene_harm__wounds=0.15).signals == set()


def test_context_sentence_sets_the_context():
    assert scene(scene_context__medical=0.3).contexts == {"medical"}


def test_backbone_without_scene_sentences_gives_none():
    others = [label for label in LABELS if not label.startswith("scene_")]
    assert score_scene(np.zeros((1, len(others))), others, SCALE, BIAS) is None


# --- NSFW levels ---

@pytest.mark.parametrize("probability, signals", [
    (0.14, set()),                      # highest seen on safe photos (a balloon dog)
    (0.85, {"nsfw_sexual_human"}),
    (0.97, {"nsfw_explicit_human"}),    # only the highest level counts
])
def test_nsfw_probability_gives_the_highest_level(probability, signals):
    assert nsfw_signals(probability) == signals


# --- settings checks ---

def test_harm_idea_with_unknown_signal_is_refused():
    data = settings.gate2.model_dump()
    data["scene"]["harm"]["abuse"]["signals"] = ["welfare_kicking"]
    with pytest.raises(ValueError, match="unknown signals"):
        Gate2Settings.model_validate(data)


def test_nsfw_level_must_be_a_d2_signal():
    data = settings.gate2.model_dump()
    data["nsfw"]["levels"] = {"item_drugs": 0.9}
    with pytest.raises(ValueError, match="not a D2 signal"):
        Gate2Settings.model_validate(data)


# --- in the pipeline ---

class FakeNsfw:
    def __init__(self, value):
        self.value = value

    def probability(self, images):
        return [self.value] * len(images)


@pytest.fixture
def nsfw_sees(monkeypatch):
    monkeypatch.setattr(settings.ml, "nsfw_model", "nsfw_vit")

    def use(probability):
        monkeypatch.setattr(orchestrator, "get_nsfw_model", lambda: FakeNsfw(probability))
    return use


def test_person_hitting_a_pet_is_a_hard_reject(tmp_path, with_model):
    with_model({**PERSONAL, "scene_harm__abuse": 0.3})

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "REJECTED"
    assert response.rejection_code == "MOD_002"
    assert response.hard_reject is True
    assert response.dimension_scores.D1_welfare == 0
    assert response.dimension_scores.D3_violence == 0


def test_animal_fighting_is_mod_003(tmp_path, with_model):
    with_model({**PERSONAL, "scene_harm__fighting": 0.3})

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.rejection_code == "MOD_003"


def test_wound_goes_to_a_human(tmp_path, with_model):
    with_model({**PERSONAL, "scene_harm__wounds": 0.3})

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "HUMAN_REVIEW"   # -50 >= review_from_deduction
    assert response.dimension_scores.D1_welfare == 50


def test_wound_with_a_cone_collar_is_treatment(tmp_path, with_model):
    with_model({**PERSONAL, "scene_harm__wounds": 0.3, "scene_context__medical": 0.3})

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "APPROVED"   # spec 7: medical context, wounds cost nothing
    assert response.dimension_scores.D1_welfare == 100


@pytest.mark.parametrize("probability, decision", [
    (0.1, "APPROVED"),
    (0.85, "HUMAN_REVIEW"),
    (0.97, "REJECTED"),
])
def test_nsfw_model_in_the_pipeline(tmp_path, with_model, nsfw_sees, probability, decision):
    with_model(PERSONAL)
    nsfw_sees(probability)

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == decision
    if decision == "REJECTED":
        assert response.rejection_code == "MOD_004"
        assert response.hard_reject is True


def test_nsfw_model_that_failed_to_load_never_approves(tmp_path, with_model, monkeypatch):
    monkeypatch.setattr(settings.ml, "nsfw_model", "nsfw_vit")
    monkeypatch.setattr(orchestrator, "get_nsfw_model", lambda: None)
    with_model(PERSONAL)

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "HUMAN_REVIEW"
    assert "nsfw_unavailable" in response.flags


def test_nsfw_runs_on_every_frame_without_a_detector(videos, with_model, nsfw_sees):
    with_model(PERSONAL)
    nsfw_sees(0.1)
    ctx = make_ctx(None, content_type="video", path=videos["mp4"])

    orchestrator.run_pipeline(ctx)

    assert sorted(ctx.nsfw) == list(range(len(ctx.prepared.frames)))
    assert len(ctx.scene) == len(ctx.prepared.frames)
    assert "nsfw" in ctx.timings


def test_nsfw_runs_only_on_the_detector_frames(videos, models, nsfw_sees):
    models(SIGLIP_SURE, [box(0.9, "dog")])
    nsfw_sees(0.1)
    ctx = make_ctx(None, content_type="video", path=videos["60s"])

    orchestrator.run_pipeline(ctx)

    assert sorted(ctx.nsfw) == sorted(ctx.detections)            # decided: speed over every frame
    assert len(ctx.nsfw) < len(ctx.prepared.frames)


def test_nudity_on_a_detector_frame_rejects_the_video(videos, models, nsfw_sees):
    models(SIGLIP_SURE, [box(0.9, "dog")])
    nsfw_sees(0.97)

    response = orchestrator.run_pipeline(make_ctx(None, content_type="video", path=videos["60s"]))

    assert response.rejection_code == "MOD_004"
