"""D5 image quality on pictures generated in the test (no models, no sample files)."""
import numpy as np
import pytest
from PIL import Image, ImageEnhance, ImageFilter

from app.core.config import Gate2QualitySettings, settings
from app.gates.quality import QualityMeasures, frame_quality, jpeg_quality, score_quality

SIDE = 1000   # original upload size given to frame_quality (above the low-resolution bar)


def detailed(size=(640, 480), seed=0) -> Image.Image:
    """A sharp, well-lit picture: random texture in the mid-tones."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(60, 200, size=(size[1] // 8, size[0] // 8, 3), dtype=np.uint8)
    small = Image.fromarray(pixels)
    return small.resize(size, Image.NEAREST).filter(ImageFilter.GaussianBlur(0.6))


def quality(picture, box=None, side=SIDE, jpeg_q=None):
    return frame_quality(picture, box, side, jpeg_q)


def test_sharp_well_lit_picture_scores_in_the_sharp_band():
    result = quality(detailed())
    assert result.issue == "sharp"
    assert 80 <= result.score <= 100


@pytest.mark.parametrize("colour", [(0, 0, 0), (255, 255, 255), (128, 128, 128)])
def test_blank_frame_scores_zero(colour):
    result = quality(Image.new("RGB", (640, 480), colour))
    assert (result.score, result.issue) == (0.0, "blank")


def test_blur_lowers_the_score_step_by_step():
    sharp = quality(detailed()).score
    slight = quality(detailed().filter(ImageFilter.GaussianBlur(1.5)))
    heavy = quality(detailed().filter(ImageFilter.GaussianBlur(6)))
    assert heavy.issue == "heavy_blur" and 10 <= heavy.score <= 30
    assert heavy.score < slight.score < sharp


def test_dark_but_sharp_photo_is_dark_not_blurred():
    result = quality(ImageEnhance.Brightness(detailed()).enhance(0.2))
    assert result.issue == "dark"
    assert 40 <= result.score <= 60


def test_burnt_out_photo_is_overexposed():
    result = quality(ImageEnhance.Brightness(detailed()).enhance(3.0))
    assert result.issue == "overexposed"


def test_white_background_is_fine_when_the_pet_box_is_well_lit():
    picture = Image.new("RGB", (640, 480), "white")
    picture.paste(detailed((200, 200)), (220, 140))
    whole = quality(picture)
    on_pet = quality(picture, box=(220, 140, 420, 340))
    assert whole.issue == "overexposed"
    assert on_pet.issue == "sharp"


def test_blurred_background_does_not_count_when_the_pet_is_sharp():
    picture = detailed().filter(ImageFilter.GaussianBlur(6))   # portrait-mode background
    picture.paste(detailed((200, 200), seed=1), (220, 140))
    assert quality(picture).issue in ("heavy_blur", "slight_blur")
    assert quality(picture, box=(220, 140, 420, 340)).issue == "sharp"


def test_tiny_pet_box_falls_back_to_the_whole_picture():
    picture = detailed()
    tiny = (10, 10, 20, 20)
    assert quality(picture, box=tiny).measures.sharpness == quality(picture).measures.sharpness


def test_low_resolution_upload_is_in_the_screenshot_band():
    result = quality(detailed(), side=240)
    assert result.issue == "screenshot"
    assert 30 <= result.score <= 50


def test_letterboxed_video_screenshot_is_caught():
    screen = Image.new("RGB", (360, 640), "black")   # portrait phone screen
    screen.paste(detailed((360, 203)), (0, 218))     # landscape video in the middle
    result = quality(screen, side=1280)
    assert result.measures.bar_fraction > 0.6
    assert result.issue == "screenshot"


def test_dark_background_is_not_a_letterbox():
    picture = detailed()
    pixels = np.asarray(picture).copy()
    pixels[:100] = np.random.default_rng(2).integers(0, 30, size=pixels[:100].shape)  # noisy dark sky
    assert quality(Image.fromarray(pixels)).measures.bar_fraction == 0


@pytest.mark.parametrize("saved_at", [5, 20, 50, 90])
def test_jpeg_quality_is_read_back_from_the_file(tmp_path, saved_at):
    path = tmp_path / "photo.jpg"
    detailed().save(path, quality=saved_at)
    assert jpeg_quality(path) == pytest.approx(saved_at, abs=3)


def test_heavily_compressed_jpeg_is_in_the_compressed_band(tmp_path):
    path = tmp_path / "photo.jpg"
    detailed().save(path, quality=5)
    result = quality(Image.open(path), jpeg_q=jpeg_quality(path))
    assert result.issue == "compressed"
    assert 20 <= result.score <= 40


def test_png_has_no_jpeg_quality(tmp_path):
    path = tmp_path / "photo.png"
    detailed().save(path)
    assert jpeg_quality(path) is None


def test_worst_problem_decides():
    m = QualityMeasures(contrast=50, brightness=30, bright_fraction=0, sharpness=1000,
                        bar_fraction=0, short_side=1000, jpeg_quality=5)
    score, issue = score_quality(m)
    assert issue == "compressed"   # 20-40 band is below the dark 40-60 band


def test_sharpness_thresholds_must_increase():
    data = settings.gate2.quality.model_dump()
    data["sharpness"]["slight_below"] = data["sharpness"]["full_at"] + 1
    with pytest.raises(ValueError, match="heavy_below < slight_below < full_at"):
        Gate2QualitySettings.model_validate(data)
