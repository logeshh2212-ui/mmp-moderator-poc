"""Gate 2 scoring and the decision engine, driven by made-up detector signals (no models)."""
import pytest
from pydantic import ValidationError

from app.core.config import Gate2Settings, settings
from app.decision.engine import decide
from app.gates.gate2 import FrameSignals, score_gate2


def frame(*signals, contexts=(), quality=100.0, relevance=100.0) -> FrameSignals:
    return FrameSignals(set(signals), set(contexts), quality, relevance)


def judge(*frames):
    result = score_gate2(list(frames))
    return result, decide(result)


# --- composite and thresholds (spec 3.2, 3.3) ---

def test_clean_pet_photo_is_approved():
    result, verdict = judge(frame(quality=90, relevance=90))
    assert result.dimensions["D1_welfare"] == 100
    assert verdict.composite_score == 98.0
    assert verdict.decision == "APPROVED"
    assert verdict.rejection_code is None


def test_composite_uses_the_spec_weights():
    # D1 70 (distress -30): 0.25*70 + 25 + 20 + 10 + 10 + 10
    _, verdict = judge(frame("welfare_distress"))
    assert verdict.composite_score == 92.5
    assert verdict.decision == "APPROVED"
    assert verdict.flags == ["welfare_distress"]


def test_dimension_never_goes_below_zero():
    result, _ = judge(frame("welfare_distress", "welfare_wounds", "welfare_graphic_dead"))
    assert result.dimensions["D1_welfare"] == 0


def test_missing_detector_means_human_review_not_a_guess():
    _, verdict = judge(frame(quality=None))
    assert verdict.decision == "HUMAN_REVIEW"
    assert verdict.composite_score is None
    assert "gate2_incomplete" in verdict.flags


def test_low_composite_rejects_with_the_highest_priority_code():
    _, verdict = judge(frame(
        "welfare_distress", "welfare_malnourished", "welfare_neglect",   # D1 15
        "nsfw_sexual_human", "violence_human_gore", "item_drugs",
    ))
    assert verdict.composite_score < settings.gate2.thresholds.review
    assert verdict.decision == "REJECTED"
    assert verdict.rejection_code == "MOD_002"   # 002 beats 004, 005, 006
    assert verdict.hard_reject is False


def test_low_composite_without_a_safety_signal_is_mod_007(monkeypatch):
    monkeypatch.setattr(settings.gate2.thresholds, "review", 79)
    _, verdict = judge(frame(quality=0, relevance=0))   # composite 80
    assert verdict.composite_score == 80
    monkeypatch.setattr(settings.gate2.thresholds, "review", 81)
    _, verdict = judge(frame(quality=0, relevance=0))
    assert verdict.rejection_code == "MOD_007"


# --- hard rejects and reason codes (spec 4, 5.2) ---

@pytest.mark.parametrize("signal, code", [
    ("welfare_fighting", "MOD_003"),
    ("welfare_abuse", "MOD_002"),
    ("welfare_trafficking", "MOD_009"),
    ("nsfw_animal", "MOD_004"),
    ("violence_slaughter", "MOD_005"),
    ("item_drug_use_on_animal", "MOD_006"),
])
def test_hard_signal_rejects_whatever_the_composite(signal, code):
    _, verdict = judge(frame(signal))
    assert verdict.decision == "REJECTED"
    assert verdict.rejection_code == code
    assert verdict.hard_reject is True


def test_exactly_one_code_by_priority():
    _, verdict = judge(frame("welfare_abuse", "nsfw_animal", "welfare_fighting"))
    assert verdict.rejection_code == "MOD_003"   # 003 > 002 > 004


# --- human review rules ---

@pytest.mark.parametrize("signal", ["item_drugs", "item_firearm", "violence_weapon_pointed",
                                    "welfare_wounds", "nsfw_sexual_human"])
def test_one_serious_signal_goes_to_a_human_even_with_a_high_composite(signal):
    _, verdict = judge(frame(signal))
    assert verdict.composite_score >= settings.gate2.thresholds.approve
    assert verdict.decision == "HUMAN_REVIEW"


@pytest.mark.parametrize("signal", ["item_alcohol", "item_tobacco", "nsfw_partial_nudity"])
def test_minor_items_are_only_flagged(signal):
    _, verdict = judge(frame(signal))
    assert verdict.decision == "APPROVED"
    assert signal in verdict.flags


@pytest.mark.parametrize("quality, relevance", [(0, 100), (100, 0), (0, 0), (60, 100)])
def test_quality_and_relevance_alone_never_hold_a_post(quality, relevance):
    # decided: D5 / D6 are reported for other uses, they only move the composite
    result, verdict = judge(frame(quality=quality, relevance=relevance))
    assert verdict.decision == "APPROVED"
    assert result.dimensions["D5_quality"] == quality
    assert result.dimensions["D6_relevance"] == relevance
    assert verdict.composite_score == 80 + (quality + relevance) / 10


def test_drug_use_on_animal_also_takes_d1():
    result, verdict = judge(frame("item_drug_use_on_animal"))
    assert result.dimensions["D4_prohibited"] == 0
    assert result.dimensions["D1_welfare"] == 0
    assert verdict.rejection_code == "MOD_006"
    assert verdict.hard_reject is True


# --- context exceptions (spec 7) ---

def test_post_surgery_wound_is_not_penalised():
    result, verdict = judge(frame("welfare_wounds", contexts=["medical"]))
    assert result.dimensions["D1_welfare"] == 100
    assert verdict.decision == "APPROVED"
    assert verdict.flags == []


def test_rescue_halves_welfare_deductions():
    result, verdict = judge(frame("welfare_wounds", "welfare_malnourished", contexts=["rescue"]))
    assert result.dimensions["D1_welfare"] == 62.5   # 100 - (50 + 25) / 2
    assert verdict.decision == "APPROVED"


def test_raw_feeding_blood_is_softened_but_checked_by_a_human():
    result, verdict = judge(frame("violence_animal_gore", contexts=["raw_feeding"]))
    assert result.dimensions["D3_violence"] == 75
    assert verdict.decision == "HUMAN_REVIEW"


def test_peaceful_memorial_is_not_penalised():
    _, verdict = judge(frame("welfare_distress", contexts=["peaceful_memorial"]))
    assert verdict.decision == "APPROVED"


def test_hunting_firearm_goes_to_a_human():
    _, verdict = judge(frame("item_firearm", contexts=["hunting"]))
    assert verdict.decision == "HUMAN_REVIEW"


@pytest.mark.parametrize("context", ["rescue", "medical", "raw_feeding", "peaceful_memorial"])
def test_context_never_softens_a_hard_reject(context):
    _, verdict = judge(frame("welfare_fighting", "welfare_abuse", contexts=[context]))
    assert verdict.decision == "REJECTED"
    assert verdict.hard_reject is True


def test_two_contexts_on_one_signal_take_the_lower_deduction():
    # medical sets wounds to 0; rescue halves D1. Lower (0) wins.
    result, _ = judge(frame("welfare_wounds", contexts=["medical", "rescue"]))
    assert result.dimensions["D1_welfare"] == 100


# --- video (spec 6.1, 6.3) ---

def test_video_one_bad_frame_is_averaged_inside_its_window():
    frames = [frame() for _ in range(10)]
    frames[5] = frame("welfare_distress")   # D1 70 in one frame
    result, _ = judge(*frames)
    assert result.dimensions["D1_welfare"] == 90   # (100 + 100 + 70) / 3


def test_video_three_bad_frames_in_a_row_govern_the_score():
    frames = [frame() for _ in range(10)]
    for i in (4, 5, 6):
        frames[i] = frame("welfare_distress")
    result, _ = judge(*frames)
    assert result.dimensions["D1_welfare"] == 70


def test_video_clean_start_then_problem_is_caught_not_averaged():
    frames = [frame() for _ in range(20)] + [frame("welfare_wounds") for _ in range(3)]
    result, verdict = judge(*frames)
    assert result.dimensions["D1_welfare"] == 50   # an average would say ~93
    assert verdict.decision == "HUMAN_REVIEW"


def test_video_one_hard_frame_rejects_the_whole_video():
    frames = [frame() for _ in range(30)]
    frames[17] = frame("violence_animal_fighting")
    _, verdict = judge(*frames)
    assert verdict.decision == "REJECTED"
    assert verdict.rejection_code == "MOD_003"


def test_video_quality_uses_the_typical_frame():
    frames = [frame(quality=80) for _ in range(9)] + [frame(quality=10)]   # one motion-blurred frame
    result, _ = judge(*frames)
    assert result.dimensions["D5_quality"] == 80


# --- settings are checked at start-up ---

def policy(**changes) -> dict:
    data = settings.gate2.model_dump()
    data.update(changes)
    return data


def test_valid_policy_loads():
    Gate2Settings.model_validate(policy())


def test_weights_must_add_up_to_one():
    weights = dict(settings.gate2.weights, D6_relevance=0.2)
    with pytest.raises(ValidationError, match="add up to 1"):
        Gate2Settings.model_validate(policy(weights=weights))


def test_every_rejection_code_needs_a_priority():
    with pytest.raises(ValidationError, match="every rejection code"):
        Gate2Settings.model_validate(policy(code_priority=settings.gate2.code_priority[:-1]))


def test_a_context_cannot_soften_a_hard_reject():
    contexts = settings.gate2.model_dump()["contexts"]
    contexts["rescue"]["deductions"] = {"welfare_abuse": 0}
    with pytest.raises(ValidationError, match="hard-reject"):
        Gate2Settings.model_validate(policy(contexts=contexts))


def test_a_context_cannot_name_an_unknown_signal():
    contexts = settings.gate2.model_dump()["contexts"]
    contexts["medical"]["deductions"] = {"welfare_wound": 0}   # typo
    with pytest.raises(ValidationError, match="unknown signals"):
        Gate2Settings.model_validate(policy(contexts=contexts))
