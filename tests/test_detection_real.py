"""OWLv2 + BioCLIP 2 with the real models. Run on purpose: uv run pytest -m models"""
from pathlib import Path

import pytest
from PIL import Image

from app.core.config import settings
from app.gates.detection import detect_animals
from app.ml.bioclip import BioclipClassifier
from app.ml.owlv2 import Owlv2Detector

pytestmark = pytest.mark.models

CAT = Path(__file__).parent / "fixtures" / "cat.jpg"
SPECIES = list(settings.gate1.species)


@pytest.fixture(scope="module")
def models():
    detector = Owlv2Detector("owlv2", labels=SPECIES)
    detector.load()
    species = BioclipClassifier("bioclip2", species=SPECIES)
    species.load()
    return detector, species


def test_cat_photo_has_a_cat_box_and_species(models):
    if not CAT.exists():
        pytest.skip("copy a real cat photo to tests/fixtures/cat.jpg")
    with Image.open(CAT) as img:
        picture = img.convert("RGB")
    picture.thumbnail((640, 640))

    [found] = detect_animals([picture], *models)

    print(f"\ncat photo: det_max {found.det_max:.2f} ({found.det_label}), species {found.species}")
    assert found.det_max >= settings.gate1.detector.score_full
    assert found.species[0] == "cat"


def test_empty_picture_has_no_animal_box(models):
    [found] = detect_animals([Image.new("RGB", (640, 480), "white")], *models)

    print(f"\nwhite picture: det_max {found.det_max:.2f}")
    assert found.det_max < settings.gate1.detector.min_box_score
    assert found.species == []
