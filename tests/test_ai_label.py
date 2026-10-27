"""AI-generated label: metadata hints first, then the model; never changes the decision."""
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from app.core.config import settings
from app.gates.ai_label import ai_label
from app.pipeline import orchestrator
from tests.test_detection import SIGLIP_SURE, box, models  # noqa: F401
from tests.test_gate1 import FOOD, make_ctx
from tests.test_gate2_pipeline import PERSONAL, with_model  # noqa: F401


# --- the rule on its own ---

def test_metadata_hint_is_certain():
    label = ai_label(["png_text:parameters"], [0.01])
    assert (label.generated, label.score) == (True, 100.0)
    assert label.flags == ["ai_generated", "ai_metadata"]


@pytest.mark.parametrize("probabilities, generated, score", [
    ([0.95], True, 95.0),
    ([0.30], False, 30.0),
    ([0.95, 0.10, 0.20], False, 20.0),   # video: the median frame, one odd frame does not decide
])
def test_model_probability_against_the_threshold(probabilities, generated, score):
    label = ai_label([], probabilities)
    assert (label.generated, label.score) == (generated, score)
    assert ("ai_generated" in label.flags) is generated


def test_no_hints_and_no_model_means_unknown():
    label = ai_label([], None)
    assert (label.generated, label.score, label.flags) == (None, None, [])


# --- in the pipeline ---

class FakeAi:
    def __init__(self, value):
        self.value = value
        self.seen = []

    def probability(self, images):
        self.seen += images
        return [self.value] * len(images)


@pytest.fixture
def ai_sees(monkeypatch):
    monkeypatch.setattr(settings.ml, "ai_model", "ai_smaug")

    def use(value):
        fake = FakeAi(value)
        monkeypatch.setattr(orchestrator, "get_ai_model", lambda: fake)
        return fake
    return use


def test_ai_picture_is_labelled_but_still_approved(tmp_path, with_model, ai_sees):
    with_model(PERSONAL)
    fake = ai_sees(0.97)

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "APPROVED"          # the label never decides
    assert (response.ai_generated, response.ai_score) == (True, 97.0)
    assert "ai_generated" in response.flags
    assert fake.seen[0].size == (64, 64)            # the original upload, not a resized frame


def test_real_photo_gets_a_low_score(tmp_path, with_model, ai_sees):
    with_model(PERSONAL)
    ai_sees(0.02)

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert (response.ai_generated, response.ai_score) == (False, 2.0)
    assert "ai_generated" not in response.flags


def test_generator_metadata_labels_without_the_model(tmp_path, with_model):
    with_model(PERSONAL)
    path = tmp_path / "sd.png"
    info = PngInfo()
    info.add_text("parameters", "a cute dog, Steps: 20, Sampler: Euler")   # Stable Diffusion writes this
    Image.new("RGB", (64, 64), "orange").save(path, pnginfo=info)

    response = orchestrator.run_pipeline(make_ctx(tmp_path, path=path))

    assert (response.ai_generated, response.ai_score) == (True, 100.0)
    assert "ai_metadata" in response.flags


def test_no_pet_upload_is_not_labelled(tmp_path, with_model, ai_sees):
    with_model(FOOD)
    fake = ai_sees(0.97)

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.rejection_code == "MOD_001"
    assert (response.ai_generated, response.ai_score) == (None, None)   # answered at Gate 1
    assert fake.seen == []


def test_metadata_hint_skips_the_model(tmp_path, with_model, ai_sees):
    with_model(PERSONAL)
    fake = ai_sees(0.01)
    path = tmp_path / "sd.png"
    info = PngInfo()
    info.add_text("parameters", "a cute dog, Steps: 20")
    Image.new("RGB", (64, 64), "orange").save(path, pnginfo=info)

    response = orchestrator.run_pipeline(make_ctx(tmp_path, path=path))

    assert fake.seen == []                       # the file already says it is AI-made
    assert (response.ai_generated, response.ai_score) == (True, 100.0)


def test_model_that_failed_to_load_falls_back_to_metadata(tmp_path, with_model, monkeypatch):
    monkeypatch.setattr(settings.ml, "ai_model", "ai_smaug")
    monkeypatch.setattr(orchestrator, "get_ai_model", lambda: None)
    with_model(PERSONAL)

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "APPROVED"
    assert response.ai_generated is None
    assert "ai_label_unavailable" in response.flags


def test_video_uses_a_few_detector_frames(videos, models, ai_sees):
    models(SIGLIP_SURE, [box(0.9, "dog")])
    fake = ai_sees(0.05)
    ctx = make_ctx(None, content_type="video", path=videos["60s"])

    orchestrator.run_pipeline(ctx)

    assert len(fake.seen) == settings.ai_label.video_frames
    assert ctx.ai_generated is False
