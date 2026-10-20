from time import perf_counter
from uuid import uuid4

from PIL import Image, ImageDraw

from app.core.config import settings
from app.pipeline import orchestrator
from app.pipeline.context import ModerationContext
from app.validation.csam import CsamResult
from app.validation.hashing import phash_image


BASE = "c3a1f0e8d2b4a596"


def flip_bits(phash: str, *bit_positions: int) -> str:
    value = int(phash, 16)
    for bit in bit_positions:
        value ^= 1 << bit
    return f"{value:016x}"


# --- the blocklist itself ---------------------------------------------------

def test_empty_blocklist_finds_nothing(blocklist):
    assert blocklist.find(BASE) is None


def test_exact_match(blocklist):
    blocklist.add(BASE, "MOD_003", hard_reject=True, source="test")

    match = blocklist.find(BASE)

    assert match.distance == 0
    assert match.entry.rejection_code == "MOD_003"
    assert match.entry.hard_reject is True
    assert match.entry.source == "test"


def test_near_match_within_max_distance(blocklist):
    blocklist.add(BASE, "MOD_003")

    match = blocklist.find(flip_bits(BASE, 0, 20, 40))  # 3 bits differ

    assert match.distance == 3


def test_seven_bits_in_seven_different_bands_still_found(blocklist, monkeypatch):
    monkeypatch.setattr(settings.blocklist, "max_distance", 7)
    blocklist.add(BASE, "MOD_003")

    # One bit in each of bands 0-6; only band 7 is still identical.
    near = flip_bits(BASE, 0, 8, 16, 24, 32, 40, 48)

    assert blocklist.find(near).distance == 7


def test_just_over_max_distance_is_not_a_match(blocklist, monkeypatch):
    monkeypatch.setattr(settings.blocklist, "max_distance", 2)
    blocklist.add(BASE, "MOD_003")

    assert blocklist.find(flip_bits(BASE, 0, 20, 40)) is None


def test_closest_entry_wins(blocklist):
    blocklist.add(BASE, "MOD_003")
    blocklist.add(flip_bits(BASE, 1), "MOD_002")

    assert blocklist.find(flip_bits(BASE, 1)).entry.rejection_code == "MOD_002"


def test_remove(blocklist):
    blocklist.add(BASE, "MOD_003")

    assert blocklist.remove(BASE) is True
    assert blocklist.find(BASE) is None
    assert blocklist.count() == 0


# --- in the pipeline --------------------------------------------------------

def scene(size=(400, 300)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((40, 40, 200, 200), fill="black")
    draw.rectangle((250, 100, 380, 280), fill="gray")
    return img


def make_ctx(tmp_path, img: Image.Image, name="x.jpg", quality=95) -> ModerationContext:
    path = tmp_path / name
    img.save(path, "JPEG", quality=quality)
    return ModerationContext(
        content_id=uuid4(),
        user_id=uuid4(),
        content_type="image",
        context="post",
        started_at=perf_counter(),
        path=path,
        sha256="a" * 64,
    )


def test_blocked_image_gets_earlier_decision(tmp_path, blocklist):
    ctx = make_ctx(tmp_path, scene())
    blocklist.add(phash_image(ctx.path), "MOD_003", hard_reject=True)

    response = orchestrator.run_pipeline(ctx)

    assert response.decision == "REJECTED"
    assert response.rejection_code == "MOD_003"
    assert response.hard_reject is True
    assert response.flags == ["known_bad_image"]


def test_resized_recompressed_copy_is_also_blocked(tmp_path, blocklist):
    original = make_ctx(tmp_path, scene(), "original.jpg")
    blocklist.add(phash_image(original.path), "MOD_003")

    copy = make_ctx(tmp_path, scene().resize((200, 150)), "copy.jpg", quality=50)
    response = orchestrator.run_pipeline(copy)

    assert response.rejection_code == "MOD_003"
    assert response.flags == ["known_bad_image"]


def test_normal_image_is_not_added(tmp_path, blocklist):
    orchestrator.run_pipeline(make_ctx(tmp_path, scene()))

    assert blocklist.count() == 0


def test_quarantined_image_is_added_automatically(tmp_path, blocklist, monkeypatch):
    class AlwaysMatches:
        def check(self, path, sha256):
            return CsamResult(checked=True, match=True, provider="fake")

    monkeypatch.setattr(orchestrator, "get_csam_checker", lambda: AlwaysMatches())
    orchestrator.run_pipeline(make_ctx(tmp_path, scene(), "first.jpg"))
    assert blocklist.count() == 1

    # The same picture again, re-saved, with a checker that finds nothing:
    monkeypatch.undo()
    monkeypatch.setattr("app.pipeline.orchestrator.get_blocklist", lambda: blocklist)
    response = orchestrator.run_pipeline(make_ctx(tmp_path, scene(), "again.jpg", quality=70))

    assert response.rejection_code == "MOD_004"
    assert response.hard_reject is True
    assert response.flags == ["known_bad_image"]


def test_bad_file_is_never_added(tmp_path, blocklist):
    ctx = make_ctx(tmp_path, scene((16, 16)))  # too small -> MOD_008

    orchestrator.run_pipeline(ctx)

    assert blocklist.count() == 0
