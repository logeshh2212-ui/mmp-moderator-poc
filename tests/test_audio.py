"""Spec 6.2: the audio track of a video (sounds + speech). D1 -20 each, never a hard reject."""
import numpy as np
import pytest

from app.core.config import Gate2Settings, settings
from app.gates.audio import audio_signals
from app.pipeline import orchestrator
from app.prepare.audio import load_audio
from tests.test_detection import SIGLIP_SURE, box, models  # noqa: F401
from tests.test_gate1 import make_ctx


# --- the rule on its own ---

def test_quiet_happy_video_has_no_audio_signals():
    result = audio_signals({"Bark": 0.9, "Music": 0.8, "Whimper (dog)": 0.05}, "good boy, who wants a treat")
    assert result.signals == set()


def test_dog_whimpering_is_distress():
    result = audio_signals({"Whimper (dog)": 0.6}, None)
    assert result.signals == {"audio_distress"}
    assert result.sounds == {"Whimper (dog)": 0.6}


def test_slap_sound_is_violence():
    assert audio_signals({"Slap, smack": 0.5}, None).signals == {"audio_violence"}


def test_listed_word_in_speech_is_abusive_speech():
    result = audio_signals(None, "watch this porn")
    assert result.signals == {"audio_abusive_speech"}
    assert result.speech_words == 1


def test_pet_words_in_speech_are_fine():
    assert audio_signals(None, "my bitch had six puppies, kutta pagal hai").signals == set()


def test_audio_signals_are_never_hard():
    data = settings.gate2.model_dump()
    data["signals"]["audio_distress"]["hard"] = True
    with pytest.raises(ValueError, match="non-hard"):
        Gate2Settings.model_validate(data)


# --- reading the soundtrack ---

def test_soundtrack_is_read_as_16khz_mono(videos):
    wave = load_audio(videos["with_audio"])
    assert wave is not None and wave.dtype == np.float32
    assert abs(len(wave) / settings.gate2.audio.sample_rate - 2.0) < 0.2    # 2 s test video
    assert np.abs(wave).max() > 0.1                                       # the 440 Hz tone


# --- in the pipeline ---

class FakeSounds:
    def __init__(self, scores):
        self.scores_ = scores

    def scores(self, wave):
        return self.scores_


class FakeSpeech:
    def __init__(self, text):
        self.text = text

    def transcribe(self, wave):
        return self.text


@pytest.fixture
def hears(monkeypatch):
    monkeypatch.setattr(settings.ml, "audio_model", "ast_audioset")
    monkeypatch.setattr(settings.ml, "speech_model", "whisper_base")

    def use(scores, text=""):
        scores = {"Speech": 0.9, **scores}   # someone talks, so Whisper runs
        monkeypatch.setattr(orchestrator, "get_audio_model", lambda: FakeSounds(scores))
        monkeypatch.setattr(orchestrator, "get_speech_model", lambda: FakeSpeech(text))
    return use


def test_distress_sound_costs_d1_20_but_still_approves(videos, models, hears):
    models(SIGLIP_SURE, [box(0.9, "dog")])
    hears({"Whimper (dog)": 0.7})

    response = orchestrator.run_pipeline(make_ctx(None, content_type="video", path=videos["with_audio"]))

    assert response.decision == "APPROVED"          # spec: audio alone never rejects, -20 < review bar
    assert response.dimension_scores.D1_welfare == 80
    assert "audio_distress" in response.flags


def test_three_audio_problems_add_up(videos, models, hears):
    models(SIGLIP_SURE, [box(0.9, "dog")])
    hears({"Whimper (dog)": 0.7, "Slap, smack": 0.6}, "porn")

    response = orchestrator.run_pipeline(make_ctx(None, content_type="video", path=videos["with_audio"]))

    assert response.dimension_scores.D1_welfare == 40
    assert response.hard_reject is False


def test_video_without_audio_skips_the_audio_models(videos, models, hears):
    models(SIGLIP_SURE, [box(0.9, "dog")])
    hears({"Whimper (dog)": 0.9})
    ctx = make_ctx(None, content_type="video", path=videos["mp4"])   # no audio track

    orchestrator.run_pipeline(ctx)

    assert ctx.audio is None and "audio" not in ctx.timings


class CountingSpeech(FakeSpeech):
    calls = 0

    def transcribe(self, wave):
        CountingSpeech.calls += 1
        return self.text


@pytest.mark.parametrize("speech_score, transcribed", [(0.1, 0), (0.8, 1)])
def test_whisper_runs_only_when_someone_talks(videos, models, monkeypatch, speech_score, transcribed):
    models(SIGLIP_SURE, [box(0.9, "dog")])
    monkeypatch.setattr(settings.ml, "audio_model", "ast_audioset")
    monkeypatch.setattr(settings.ml, "speech_model", "whisper_base")
    monkeypatch.setattr(orchestrator, "get_audio_model", lambda: FakeSounds({"Music": 0.8, "Speech": speech_score}))
    monkeypatch.setattr(orchestrator, "get_speech_model", lambda: CountingSpeech("hello"))
    CountingSpeech.calls = 0

    orchestrator.run_pipeline(make_ctx(None, content_type="video", path=videos["with_audio"]))

    assert CountingSpeech.calls == transcribed


def test_missing_audio_model_is_flagged_not_blocking(videos, models, monkeypatch):
    models(SIGLIP_SURE, [box(0.9, "dog")])
    monkeypatch.setattr(settings.ml, "audio_model", "ast_audioset")
    monkeypatch.setattr(orchestrator, "get_audio_model", lambda: None)

    response = orchestrator.run_pipeline(make_ctx(None, content_type="video", path=videos["with_audio"]))

    assert "audio_unavailable" in response.flags
    assert response.decision == "APPROVED"
