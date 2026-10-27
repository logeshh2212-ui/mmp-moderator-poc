"""D4 prohibited items: OWLv2 item boxes -> Gate 2 signals (fake detector, no models)."""
import pytest

from app.core.config import Gate2Settings, settings
from app.gates.detection import detect_animals
from app.gates.items import item_signals
from app.ml.owlv2 import Box
from app.pipeline import orchestrator
from tests.test_detection import SIGLIP_SURE, FakeDetector, FakeSpecies, box, make_ctx, models, pictures  # noqa: F401


def item(score, label):
    return Box(score, label, (60.0, 60.0, 90.0, 90.0), animal=False, item=True)


DOG = box(0.9, "dog")


# --- item queries -> signals ---

@pytest.mark.parametrize("found, signals", [
    ({"gun": 0.5}, {"item_firearm"}),
    ({"rifle": 0.45, "beer bottle": 0.6}, {"item_firearm", "item_alcohol"}),
    ({"cigarette": 0.45}, {"item_tobacco"}),
    ({"bong": 0.5}, {"item_drugs"}),
    ({"machete": 0.5}, {"item_weapon_knife"}),
    ({"gun": 0.2}, set()),                  # below gate2.items.min_score
    ({"machete": 0.36}, set()),             # measured false alarm on a pet photo
    ({"toy gun": 0.9}, set()),              # harmless look-alike
    ({"kitchen knife": 0.9}, set()),        # spec 4.4: incidental kitchen knife = ignore
    ({}, set()),
])
def test_item_boxes_become_signals(found, signals):
    assert item_signals(found) == signals


def test_every_item_query_maps_to_a_d4_signal_or_nothing():
    for signal in settings.gate2.items.queries.values():
        assert signal is None or settings.gate2.signals[signal].dimension == "D4_prohibited"


def test_item_query_with_a_non_d4_signal_is_refused():
    data = settings.gate2.model_dump()
    data["items"]["queries"]["gun"] = "welfare_abuse"
    with pytest.raises(ValueError, match="not a D4 signal"):
        Gate2Settings.model_validate(data)


# --- detection keeps item boxes apart from animals and look-alikes ---

def test_item_box_is_kept_and_does_not_count_as_an_animal():
    [found] = detect_animals(pictures(), FakeDetector([item(0.7, "gun")]), FakeSpecies("dog"))

    assert found.det_max == 0.0           # Gate 1 still sees no animal
    assert found.lookalike is None        # an item is not a look-alike
    assert found.items == {"gun": 0.7}


def test_best_score_per_item_query_is_kept():
    boxes = [DOG, item(0.8, "gun"), item(0.4, "gun"), item(0.5, "cigar")]
    [found] = detect_animals(pictures(), FakeDetector(boxes), FakeSpecies("dog"))

    assert found.det_max == 0.9
    assert found.items == {"gun": 0.8, "cigar": 0.5}


# --- in the pipeline ---

def test_dog_with_a_gun_goes_to_a_human(tmp_path, models):
    models(SIGLIP_SURE, [DOG, item(0.6, "gun")], ("dog", 0.9))

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "HUMAN_REVIEW"
    assert response.dimension_scores.D4_prohibited == 40
    assert "item_firearm" in response.flags
    assert "gate2_review" in response.flags


def test_dog_with_a_toy_gun_is_approved(tmp_path, models):
    models(SIGLIP_SURE, [DOG, item(0.8, "toy gun")], ("dog", 0.9))

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "APPROVED"
    assert response.dimension_scores.D4_prohibited == 100


def test_dog_next_to_a_beer_is_approved_with_a_flag(tmp_path, models):
    models(SIGLIP_SURE, [DOG, item(0.6, "beer bottle")], ("dog", 0.9))

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "APPROVED"   # spec 4.4: accept, minor flag only
    assert response.dimension_scores.D4_prohibited == 90
    assert "item_alcohol" in response.flags


def test_gun_without_any_animal_is_still_mod_001(tmp_path, models):
    models(SIGLIP_SURE, [item(0.9, "gun")], ("dog", 0.9))

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.rejection_code == "MOD_001"   # Gate 1 decides first; no Gate 2
