"""Text on images (OCR) and the shared word lists (spec 4.2 explicit text, section 7 memes)."""
import pytest

from app.core.config import Gate2Settings, settings
from app.gates.text import text_hits, text_signals, tokens
from app.pipeline import orchestrator
from tests.test_detection import SIGLIP_SURE, box, models  # noqa: F401
from tests.test_gate1 import make_ctx
from tests.test_gate2_pipeline import PERSONAL, with_model  # noqa: F401


# --- word matching ---

@pytest.mark.parametrize("text", [
    "My good boy Max!",
    "Look at this bitch and her 6 puppies",      # female dog
    "pussycat nap time", "the cock crowed at dawn", "a blue tit on the feeder",
    "What sex is my kitten?", "puppies on mum's nipples", "stop humping the pillow",
    "kutta pagal hai", "meri kuttiya ke bachche",  # Hinglish: dog, female dog
    "Sussex spaniel", "Essex dog show",            # a listed word inside another word
])
def test_normal_pet_text_has_no_hits(text):
    assert text_hits(text) == []


@pytest.mark.parametrize("text, hit", [
    ("FREE PORN click here", "porn"),
    ("xxx videos", "xxx"),
    ("nude pics in bio!!", "nude"),            # punctuation around the word
    ("NSFW", "nsfw"),
])
def test_listed_words_are_found(text, hit):
    assert hit in text_hits(text)


def test_listed_phrase_is_found_as_a_whole():
    assert text_hits("doggy style") == ["doggy style"]
    assert text_hits("doggy styled haircut") == []   # not the whole phrase


def test_devanagari_vowel_signs_stay_inside_the_word():
    assert tokens("कुत्ता, बिल्ली!") == ["कुत्ता", "बिल्ली"]


def test_hit_gives_the_text_signal():
    assert text_signals("porn") == {settings.gate2.text.signal}
    assert text_signals("good dog") == set()


def test_text_signal_must_exist():
    data = settings.gate2.model_dump()
    data["text"]["signal"] = "nsfw_rude_words"
    with pytest.raises(ValueError, match="unknown signal"):
        Gate2Settings.model_validate(data)


def test_missing_word_list_file_is_refused():
    data = settings.gate2.model_dump()
    data["text"]["blocklists"] = ["config/text/no_such_list.txt"]
    with pytest.raises(ValueError, match="not found"):
        Gate2Settings.model_validate(data)


# --- in the pipeline ---

class FakeOcr:
    def __init__(self, text):
        self.text = text
        self.seen = 0

    def read(self, images):
        self.seen += len(images)
        return [self.text] * len(images)


@pytest.fixture
def ocr_reads(monkeypatch):
    monkeypatch.setattr(settings.ml, "ocr", True)

    def use(text):
        fake = FakeOcr(text)
        monkeypatch.setattr(orchestrator, "get_ocr", lambda: fake)
        return fake
    return use


def test_meme_with_normal_caption_is_approved(tmp_path, with_model, ocr_reads):
    with_model(PERSONAL)
    ocr_reads("when the hooman says no treats")

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "APPROVED"
    assert response.dimension_scores.D2_nsfw == 100


def test_explicit_text_overlay_goes_to_a_human(tmp_path, with_model, ocr_reads):
    with_model(PERSONAL)
    ocr_reads("FREE PORN link in bio")

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "HUMAN_REVIEW"     # spec: -40, not a hard reject
    assert response.dimension_scores.D2_nsfw == 60
    assert "nsfw_explicit_text" in response.flags


def test_text_is_read_on_a_few_video_frames(videos, models, ocr_reads):
    models(SIGLIP_SURE, [box(0.9, "dog")])
    fake = ocr_reads("good boy")
    ctx = make_ctx(None, content_type="video", path=videos["60s"])

    orchestrator.run_pipeline(ctx)

    assert fake.seen == settings.gate2.text.ocr.video_frames
    assert "ocr" in ctx.timings


def test_ocr_that_failed_to_load_is_flagged_but_does_not_block(tmp_path, with_model, monkeypatch):
    monkeypatch.setattr(settings.ml, "ocr", True)
    monkeypatch.setattr(orchestrator, "get_ocr", lambda: None)
    with_model(PERSONAL)

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "APPROVED"
    assert "ocr_unavailable" in response.flags
