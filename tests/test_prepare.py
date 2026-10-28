import shutil
import subprocess
from io import BytesIO
from time import perf_counter
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from app.core.config import settings
from app.pipeline import orchestrator
from app.pipeline.context import ModerationContext
from app.prepare.images import prepare_image
from app.prepare.video import decode, fit, prepare_video, sample_timestamps, select_expression
from app.validation.errors import ProcessingTimeout


def save(tmp_path, img, name, **kwargs):
    path = tmp_path / name
    img.save(path, **kwargs)
    return path


# --- images ------------------------------------------------------------------

def test_image_sizes_keep_aspect_ratio(tmp_path):
    [frame] = prepare_image(save(tmp_path, Image.new("RGB", (400, 300), "orange"), "a.jpg")).frames

    assert frame.timestamp is None
    assert frame.detector.size == (640, 480)
    assert frame.backbone.size == (384, 288)
    assert frame.detector.mode == frame.backbone.mode == "RGB"


def test_large_jpeg_is_reduced(tmp_path):
    path = save(tmp_path, Image.new("RGB", (4000, 3000), "orange"), "big.jpg")

    assert prepare_image(path).frames[0].detector.size == (640, 480)


def test_small_image_is_scaled_up(tmp_path):
    [frame] = prepare_image(save(tmp_path, Image.new("RGB", (32, 32), "orange"), "s.png")).frames

    assert frame.detector.size == (640, 640)
    assert frame.backbone.size == (384, 384)


def test_exif_rotation_is_applied(tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways
    path = save(tmp_path, Image.new("RGB", (200, 100), "orange"), "r.jpg", exif=exif)

    assert prepare_image(path).frames[0].detector.size == (320, 640)  # now portrait


def test_transparency_becomes_white(tmp_path):
    path = save(tmp_path, Image.new("RGBA", (100, 100), (0, 0, 0, 0)), "t.png")

    assert prepare_image(path).frames[0].detector.getpixel((0, 0)) == (255, 255, 255)


def test_animated_image_uses_first_frame(tmp_path):
    frames = [Image.new("RGB", (64, 64), color) for color in ("red", "blue")]
    path = tmp_path / "a.webp"
    frames[0].save(path, "WEBP", save_all=True, append_images=frames[1:], lossless=True)

    r, g, b = prepare_image(path).frames[0].detector.getpixel((32, 32))
    assert r > 200 and b < 50


# --- choosing video frame times ----------------------------------------------

def test_short_video_every_half_second_plus_middle_and_last():
    assert sample_timestamps(3.0) == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.9]


def test_long_video_one_frame_per_second():
    times = sample_timestamps(30.0)

    assert times[:3] == [0.0, 1.0, 2.0]
    assert 15.0 in times
    assert times[-1] == 29.9
    assert len(times) == 31


def test_longest_allowed_video_stays_within_frame_limit():
    assert len(sample_timestamps(60.5)) <= settings.prepare.max_video_frames


def test_frame_limit_keeps_first_and_last(monkeypatch):
    monkeypatch.setattr(settings.prepare, "max_video_frames", 10)
    times = sample_timestamps(60.0)

    assert len(times) == 10
    assert times[0] == 0.0
    assert times[-1] == 59.9


# --- videos ------------------------------------------------------------------

def test_short_video_frames_and_times(videos):
    media = prepare_video(videos["mp4"], 2.0)
    expected = sample_timestamps(2.0)

    assert len(media.frames) == len(expected)
    for frame, wanted in zip(media.frames, expected):
        assert frame.timestamp == pytest.approx(wanted, abs=0.11)
    assert media.frames[0].detector.size == (640, 480)
    assert media.frames[0].backbone.size == (384, 288)


def test_60_second_video(videos):
    media = prepare_video(videos["60s"], 60.0)

    assert len(media.frames) == len(sample_timestamps(60.0))


def test_rotated_phone_video_comes_out_upright(videos, tmp_path):
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        [shutil.which("ffmpeg"), "-y", "-v", "error", "-display_rotation", "90",
         "-i", str(videos["mp4"]), "-c", "copy", str(rotated)],
        check=True,
        timeout=60,
    )

    width, height = prepare_video(rotated, 2.0).frames[0].detector.size
    assert width < height


def test_two_pieces_give_exactly_the_same_frames(videos):
    times = sample_timestamps(60.0)
    whole = decode(videos["60s"], 60.0, times, None, 1)
    halves = decode(videos["60s"], 60.0, times, None, 2)

    assert len(halves) == len(whole) == len(times)
    for (png_a, t_a), (png_b, t_b) in zip(whole, halves):
        assert t_b == pytest.approx(t_a, abs=0.001)
        assert np.array_equal(np.asarray(Image.open(BytesIO(png_a))), np.asarray(Image.open(BytesIO(png_b))))


def test_gpu_decode_failure_falls_back_to_the_cpu(videos, monkeypatch, caplog):
    monkeypatch.setattr(settings.prepare, "ffmpeg_hwaccel", "cuda")
    if shutil.which("nvidia-smi"):
        pytest.skip("this machine has an NVIDIA GPU: the GPU path would succeed")

    media = prepare_video(videos["60s"], 60.0, 160, 120)

    assert len(media.frames) == len(sample_timestamps(60.0))
    assert "decoding on the CPU instead" in caplog.text


def test_select_expression_adds_the_first_frame_only_for_time_zero():
    assert select_expression([0.0, 1.0]).startswith("isnan(prev_t)")
    assert "isnan" not in select_expression([30.0, 31.0])


@pytest.mark.parametrize("w, h, expected", [(1920, 1080, (640, 360)), (1080, 1920, (360, 640)), (320, 240, (320, 240))])
def test_gpu_resize_keeps_the_shape_inside_the_detector_size(w, h, expected):
    assert fit(w, h, 640) == expected


def test_frame_sampling_timeout(videos, monkeypatch):
    monkeypatch.setattr(settings.timeouts, "frame_sample_seconds", 0.001)

    with pytest.raises(ProcessingTimeout):
        prepare_video(videos["mp4"], 2.0)


# --- in the pipeline ---------------------------------------------------------

def make_ctx(path, content_type) -> ModerationContext:
    return ModerationContext(
        content_id=uuid4(),
        user_id=uuid4(),
        content_type=content_type,
        context="post",
        started_at=perf_counter(),
        path=path,
        sha256="a" * 64,
    )


def test_pipeline_prepares_an_image(tmp_path):
    ctx = make_ctx(save(tmp_path, Image.new("RGB", (400, 300), "orange"), "x.jpg"), "image")

    response = orchestrator.run_pipeline(ctx)

    assert response.flags == ["validation_passed"]
    assert len(ctx.prepared.frames) == 1


def test_pipeline_prepares_a_video(videos):
    ctx = make_ctx(videos["mp4"], "video")

    response = orchestrator.run_pipeline(ctx)

    assert response.flags == ["validation_passed"]
    assert len(ctx.prepared.frames) == len(sample_timestamps(ctx.validation.video.duration_seconds))
