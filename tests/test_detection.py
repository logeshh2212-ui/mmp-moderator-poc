"""Gate 1 detector branch (OWLv2 + BioCLIP) with fake models."""
import logging
from time import perf_counter
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from app.core.config import settings
from app.gates.detection import detect_animals, detector_frame_indices
from app.gates.gate1 import FrameDetections, detector_score, fuse, run_gate1
from app.ml.owlv2 import Box
from app.pipeline import orchestrator
from app.pipeline.context import ModerationContext
from tests.test_gate1 import LABELS, SCALE, FakeBackbone, row

SIGLIP_SURE = {"animal__birds_SLASH_a bird in a cage": 0.3, "photo__": 0.3}   # SigLIP: 100
SIGLIP_NO = {"no_pet__a room interior": 0.3, "photo__": 0.3}                   # SigLIP: ~0


class FakeDetector:
    def __init__(self, boxes_per_picture):
        self.boxes = boxes_per_picture

    def detect(self, images):
        return [list(self.boxes) for _ in images]


class FakeSpecies:
    def __init__(self, name, confidence=0.9):
        self.name, self.confidence = name, confidence
        self.calls = 0

    def classify(self, crops):
        self.calls += 1
        return [(self.name, self.confidence) for _ in crops]


def box(score, label="dog"):
    return Box(score, label, (10.0, 10.0, 50.0, 50.0))


def gate1(siglip_values, det_max, species=(), is_video=False, confidence=0.9):
    detections = [FrameDetections(det_max, "dog" if det_max else None, list(species), confidence)]
    return run_gate1(row(**siglip_values)[None, :], LABELS, SCALE, is_video, detections)


# --- fusion -------------------------------------------------------------------

def test_detector_score_maps_floor_and_full():
    assert detector_score(0.05) == 0
    assert detector_score(0.10) == 0
    assert detector_score(0.30) == pytest.approx(50)
    assert detector_score(0.50) == 100
    assert detector_score(0.95) == 100


def test_fusion_weights_come_from_settings():
    assert fuse(100, 0.5) == pytest.approx(100)
    assert fuse(100, 0.0) == pytest.approx(30)   # 0.7 x 0 + 0.3 x 100
    assert fuse(0, 0.5) == pytest.approx(70)     # 0.7 x 100 + 0.3 x 0


def test_accessory_only_photo_fails():
    # Empty cage: SigLIP thinks "a bird in a cage" (100), the detector finds no animal box.
    result = gate1(SIGLIP_SURE, det_max=0.05)

    assert result.outcome == "fail"
    assert result.pet_score == 30
    assert result.species == []


def test_animal_found_by_both_branches_passes():
    result = gate1(SIGLIP_SURE, det_max=0.9, species=["budgerigar"])

    assert result.outcome == "pass"
    assert result.pet_score == 100
    assert result.species == ["budgerigar"]


def test_detector_rescues_a_pet_siglip_misses():
    # Spec 9.2: pet small in the background, person in front.
    result = gate1(SIGLIP_NO, det_max=0.9, species=["dog"])

    assert result.pet_score >= 70
    assert result.outcome in ("pass", "borderline")  # never rejected


def test_species_category_sets_the_insect_bar():
    result = gate1(SIGLIP_SURE, det_max=0.35, species=["beetle"])  # 0.7 x 62.5 + 30 = 74

    assert result.frames[0].insect is True
    assert result.threshold == settings.gate1.thresholds.pass_insect
    assert result.outcome == "pass"


def test_species_category_sets_wild_context():
    result = gate1({**SIGLIP_SURE, "wild__": 0.4}, det_max=0.9, species=["dolphin"])

    assert result.outcome == "pass"
    assert result.wild is True


def test_without_detector_siglip_alone_decides():
    result = run_gate1(row(**SIGLIP_SURE)[None, :], LABELS, SCALE, False, None)

    assert result.pet_score == 100
    assert result.frames[0].det_max is None


# --- detect_animals -----------------------------------------------------------

def pictures(n=1):
    return [Image.new("RGB", (64, 64), "orange") for _ in range(n)]


def test_no_boxes_means_det_max_zero():
    [found] = detect_animals(pictures(), FakeDetector([]), FakeSpecies("dog"))

    assert (found.det_max, found.det_label, found.species) == (0.0, None, [])
    assert found.box is None


def test_best_animal_box_is_kept_for_d5():
    [found] = detect_animals(pictures(), FakeDetector([box(0.8, "dog")]), FakeSpecies("dog"))

    assert found.box == (10.0, 10.0, 50.0, 50.0)


def test_species_model_names_the_good_boxes():
    species_model = FakeSpecies("goldfish")
    [found] = detect_animals(pictures(), FakeDetector([box(0.8, "fish")]), species_model)

    assert found.det_max == 0.8
    assert found.species == ["goldfish"]


def test_unsure_species_model_falls_back_to_detector_name():
    [found] = detect_animals(pictures(), FakeDetector([box(0.8, "snail")]), FakeSpecies("horse", 0.2))

    assert found.species == ["snail"]


def test_weak_boxes_are_not_named():
    species_model = FakeSpecies("dog")
    [found] = detect_animals(pictures(), FakeDetector([box(0.2, "dog")]), species_model)

    assert found.det_max == 0.2
    assert found.species == []
    assert species_model.calls == 0  # no crop worth naming


def test_several_animals_give_several_species():
    class TwoSpecies:
        def classify(self, crops):
            return [("dog", 0.9), ("cat", 0.9)][: len(crops)]

    [found] = detect_animals(pictures(), FakeDetector([box(0.9, "dog"), box(0.8, "cat")]), TwoSpecies())

    assert found.species == ["dog", "cat"]


def test_detector_frames_for_images_and_videos(monkeypatch):
    monkeypatch.setattr(settings.gate1.detector, "video_frames", 8)

    assert detector_frame_indices(1, is_video=False) == [0]
    assert detector_frame_indices(5, is_video=True) == [0, 1, 2, 3, 4]
    indices = detector_frame_indices(61, is_video=True)
    assert len(indices) == 8
    assert indices[0] == 0 and indices[-1] == 60


# --- in the pipeline ----------------------------------------------------------

def make_ctx(tmp_path) -> ModerationContext:
    path = tmp_path / "x.jpg"
    Image.new("RGB", (64, 64), "orange").save(path, "JPEG")
    return ModerationContext(
        content_id=uuid4(), user_id=uuid4(), content_type="image", context="post",
        started_at=perf_counter(), path=path, sha256="a" * 64,
    )


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setattr(settings.ml, "backbone", "siglip2_base")
    monkeypatch.setattr(settings.ml, "detector", "owlv2")
    monkeypatch.setattr(settings.ml, "species_model", "bioclip2")

    def use(siglip_values, boxes, species=("dog", 0.9)):
        monkeypatch.setattr(orchestrator, "get_backbone", lambda: FakeBackbone(siglip_values))
        monkeypatch.setattr(orchestrator, "get_detector", lambda: FakeDetector(boxes))
        monkeypatch.setattr(orchestrator, "get_species_model", lambda: FakeSpecies(*species))
    return use


def test_pipeline_approves_with_species(tmp_path, models):
    models(SIGLIP_SURE, [box(0.9, "budgerigar")], ("budgerigar", 0.8))

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "APPROVED"
    assert response.species_detected == ["budgerigar"]


def test_pipeline_rejects_accessory_only(tmp_path, models):
    models(SIGLIP_SURE, [])

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "REJECTED"
    assert response.rejection_code == "MOD_001"
    assert response.species_detected == []


def test_pipeline_detector_not_loaded_is_flagged(tmp_path, models, monkeypatch):
    models(SIGLIP_SURE, [])
    monkeypatch.setattr(orchestrator, "get_detector", lambda: None)

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "HUMAN_REVIEW"
    assert response.flags == ["validation_passed", "gate1_unavailable"]


def test_pipeline_logs_branches_and_timings(tmp_path, models, caplog):
    models(SIGLIP_SURE, [box(0.9, "budgerigar")], ("budgerigar", 0.8))

    with caplog.at_level(logging.INFO):
        orchestrator.run_pipeline(make_ctx(tmp_path))

    gate1_line = next(r.getMessage() for r in caplog.records if r.name == "mmp.gate1")
    result_line = next(r.getMessage() for r in caplog.records if r.name == "mmp.moderation")
    assert "detector=budgerigar (0.90)" in gate1_line
    assert "species=budgerigar" in gate1_line
    assert "siglip=100" in gate1_line
    for stage in ("validation", "safety", "prepare", "siglip", "detector+species", "gate1"):
        assert f"{stage}:" in result_line


# --- hard-case fixes (gate1-fused-0.4) ----------------------------------------

def test_unsure_species_does_not_lower_the_bar():
    # BioCLIP said "tarantula" at 0.35 for a seascape: that must not bring in the insect bar.
    result = gate1(SIGLIP_SURE, det_max=0.35, species=["tarantula"], confidence=0.35)

    assert result.frames[0].insect is False
    assert result.threshold == settings.gate1.thresholds.pass_photo
    assert result.outcome == "borderline"  # 74 < 75


def test_marine_mammal_always_goes_to_review():
    # Dolphinarium with trainers: owner / domestic context, still never a pet (decided).
    result = gate1({**SIGLIP_SURE, "domestic__": 0.4}, det_max=0.9, species=["dolphin"])

    assert result.outcome == "pass"
    assert result.wild is True


def test_lookalike_boxes_never_count_as_animals():
    figurine = Box(0.9, "ceramic figurine", (10.0, 10.0, 50.0, 50.0), animal=False)

    [found] = detect_animals(pictures(), FakeDetector([figurine]), FakeSpecies("cat"))

    assert found.det_max == 0.0
    assert found.species == []
    assert found.lookalike == "ceramic figurine (0.90)"


def test_animal_next_to_a_lookalike_still_counts():
    toy = Box(0.95, "plush toy", (0.0, 0.0, 30.0, 30.0), animal=False)
    dog = Box(0.8, "dog", (30.0, 30.0, 60.0, 60.0))

    [found] = detect_animals(pictures(), FakeDetector([toy, dog]), FakeSpecies("dog"))

    assert found.det_max == 0.8  # a dog playing with a toy is still a dog
    assert found.species == ["dog"]
    assert found.species_confidence == 0.9


def test_pipeline_rejects_a_figurine(tmp_path, models):
    models(SIGLIP_SURE, [Box(0.9, "statue", (10.0, 10.0, 50.0, 50.0), animal=False)])

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "REJECTED"
    assert response.rejection_code == "MOD_001"
