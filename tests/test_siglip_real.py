"""Real SigLIP 2 tests. Run on purpose: uv run pytest -m models"""
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.ml.siglip import SiglipEmbedder

pytestmark = pytest.mark.models

CAT = Path(__file__).parent / "fixtures" / "cat.jpg"


@pytest.fixture(scope="module")
def siglip():
    model = SiglipEmbedder("siglip2_base", prompts={
        "cat": "a photo of a cat",
        "car": "a photo of a car",
        "no_animal": "a photo with no animal in it",
    })
    model.load()
    return model


def picture(color="orange"):
    return Image.new("RGB", (384, 288), color)


def test_server_mode_is_offline():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_embedding_size_and_length(siglip):
    embeds = siglip.embed_images([picture()])

    assert embeds.shape == (1, 768)
    assert np.linalg.norm(embeds[0]) == pytest.approx(1.0, abs=1e-4)


def test_same_picture_same_numbers(siglip):
    a, b = siglip.embed_images([picture()]), siglip.embed_images([picture()])

    np.testing.assert_allclose(a, b, atol=1e-5)


def test_alone_or_in_a_batch_gives_the_same_result(siglip):
    alone = siglip.embed_images([picture("orange")])
    batch = siglip.embed_images([picture("blue"), picture("orange"), picture("green")])

    np.testing.assert_allclose(alone[0], batch[1], atol=1e-4)


def test_text_half_is_gone(siglip):
    assert siglip.vision is not None
    assert not hasattr(siglip, "model")


def test_cat_photo_scores_cat_highest(siglip):
    if not CAT.exists():
        pytest.skip("copy a real cat photo to tests/fixtures/cat.jpg")
    with Image.open(CAT) as img:
        embeds = siglip.embed_images([img.convert("RGB")])

    probs = siglip.probabilities(embeds)

    assert probs["cat"][0] > probs["car"][0]
    assert probs["cat"][0] > probs["no_animal"][0]
