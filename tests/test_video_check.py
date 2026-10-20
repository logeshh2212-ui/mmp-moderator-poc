import pytest

from app.core.config import settings
from app.validation.errors import ProcessingTimeout, ValidationFailure
from app.validation.video_check import check_video, parse_fps


@pytest.mark.parametrize("name", ["mp4", "mov", "avi"])
def test_valid_videos_pass(videos, name):
    info = check_video(videos[name], name)

    assert (info.width, info.height) == (160, 120)
    assert info.fps == 10
    assert info.duration_seconds == pytest.approx(2, abs=0.2)
    assert info.has_audio is False
    assert info.audio_codec is None
    assert info.first_frame_png.startswith(b"\x89PNG")


def test_video_with_audio_records_audio(videos):
    info = check_video(videos["with_audio"], "mp4")

    assert info.has_audio is True
    assert info.audio_codec == "aac"
    assert info.video_codec == "h264"


def test_60_second_video_passes(videos):
    assert check_video(videos["60s"], "mp4").duration_seconds <= 60.5


def test_61_second_video_rejected(videos):
    with pytest.raises(ValidationFailure) as exc:
        check_video(videos["61s"], "mp4")
    assert exc.value.flag == "duration_exceeded"


def test_audio_only_rejected(videos):
    with pytest.raises(ValidationFailure) as exc:
        check_video(videos["audio_only"], "mp4")
    assert exc.value.flag == "no_video_stream"


def test_corrupt_video_rejected(videos):
    with pytest.raises(ValidationFailure) as exc:
        check_video(videos["corrupt"], "mp4")
    assert exc.value.flag == "corrupt_file"


def test_container_must_match_detected_format(videos):
    with pytest.raises(ValidationFailure) as exc:
        check_video(videos["avi"], "mp4")
    assert exc.value.flag == "type_mismatch"


def test_duration_limit_comes_from_settings(videos, monkeypatch):
    monkeypatch.setattr(settings.limits, "max_video_duration_seconds", 1)
    monkeypatch.setattr(settings.limits, "duration_tolerance_seconds", 0.5)

    with pytest.raises(ValidationFailure) as exc:
        check_video(videos["mp4"], "mp4")  # 2 s > 1.5 s
    assert exc.value.flag == "duration_exceeded"


def test_ffprobe_timeout_raises_processing_timeout(videos, monkeypatch):
    monkeypatch.setattr(settings.timeouts, "ffprobe_seconds", 0.001)

    with pytest.raises(ProcessingTimeout):
        check_video(videos["mp4"], "mp4")


@pytest.mark.parametrize("rate, expected", [
    ("30/1", 30.0),
    ("30000/1001", 29.97),
    ("0/0", None),
    (None, None),
    ("abc", None),
])
def test_parse_fps(rate, expected):
    assert parse_fps(rate) == expected
