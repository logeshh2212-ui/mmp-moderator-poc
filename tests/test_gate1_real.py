"""Gate 1 with the real SigLIP 2 base model. Run on purpose: uv run pytest -m models"""
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.gates.gate1 import gate1_prompts, run_gate1
from app.ml.siglip import SiglipEmbedder

pytestmark = pytest.mark.models

CAT = Path(__file__).parent / "fixtures" / "cat.jpg"


@pytest.fixture(scope="module")
def siglip():
    model = SiglipEmbedder("siglip2_base", prompts=gate1_prompts())
    model.load()
    return model


def gate1(siglip, img):
    embeds = siglip.embed_images([img.convert("RGB")])
    return run_gate1(siglip.similarities(embeds), siglip.labels, siglip.scale, is_video=False)


def test_cat_photo_passes_as_a_cat(siglip):
    if not CAT.exists():
        pytest.skip("copy a real cat photo to tests/fixtures/cat.jpg")
    with Image.open(CAT) as img:
        result = gate1(siglip, img)

    print(f"\ncat photo: pet_score {result.pet_score}, best {result.frames[0].best_animal}")
    assert result.outcome == "pass"
    assert result.frames[0].best_animal.startswith("cats/")


def test_empty_white_picture_fails(siglip):
    result = gate1(siglip, Image.new("RGB", (384, 288), "white"))

    print(f"\nwhite picture: pet_score {result.pet_score}, best no-pet {result.frames[0].best_no_pet}")
    assert result.outcome == "fail"


def test_shapes_without_animal_fail(siglip):
    img = Image.new("RGB", (384, 288), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 200, 200), fill="navy")
    draw.ellipse((220, 60, 360, 240), fill="orange")

    result = gate1(siglip, img)

    print(f"\nshapes: pet_score {result.pet_score}, best no-pet {result.frames[0].best_no_pet}")
    assert result.outcome == "fail"
