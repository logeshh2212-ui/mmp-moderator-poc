from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import CONFIG_FILE, load_settings


def write_settings(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def real_settings() -> str:
    return CONFIG_FILE.read_text(encoding="utf-8")


def test_real_settings_file_loads():
    assert load_settings().model_version


def test_unknown_key_in_a_section_is_rejected(tmp_path):
    text = real_settings().replace(
        "  min_image_side_pixels:", "  min_image_side_pixles: 32\n  min_image_side_pixels:"
    )

    with pytest.raises(ValidationError) as exc:
        load_settings(write_settings(tmp_path, text))
    assert "min_image_side_pixles" in str(exc.value)


def test_unknown_top_level_section_is_rejected(tmp_path):
    text = real_settings() + "\nsomething_new:\n  value: 1\n"

    with pytest.raises(ValidationError) as exc:
        load_settings(write_settings(tmp_path, text))
    assert "something_new" in str(exc.value)


def test_duplicate_section_is_rejected(tmp_path):
    text = real_settings() + "\nlogging:\n  level: DEBUG\n"

    with pytest.raises(ValueError) as exc:
        load_settings(write_settings(tmp_path, text))
    assert "Duplicate key 'logging'" in str(exc.value)


def test_missing_key_is_rejected(tmp_path):
    text = real_settings().replace("  socket_timeout_seconds: 2\n", "")

    with pytest.raises(ValidationError) as exc:
        load_settings(write_settings(tmp_path, text))
    assert "socket_timeout_seconds" in str(exc.value)


def test_backbone_must_be_a_listed_model(tmp_path):
    text = real_settings().replace("backbone: siglip2_base", "backbone: siglip3_huge")

    with pytest.raises(ValidationError) as exc:
        load_settings(write_settings(tmp_path, text))
    assert "siglip3_huge" in str(exc.value)

def test_float16_on_cpu_is_rejected(tmp_path):
    text = real_settings().replace("dtype: float32", "dtype: float16")

    with pytest.raises(ValidationError) as exc:
        load_settings(write_settings(tmp_path, text))
    assert "float16" in str(exc.value)


def test_device_can_be_switched_by_environment(monkeypatch):
    monkeypatch.setenv("ML_DEVICE", "cuda")
    monkeypatch.setenv("ML_DTYPE", "float16")

    loaded = load_settings()

    assert (loaded.ml.device, loaded.ml.dtype) == ("cuda", "float16")



def test_detector_must_be_a_listed_model(tmp_path):
    text = real_settings().replace("detector: owlv2 ", "detector: yolo99 ")

    with pytest.raises(ValidationError) as exc:
        load_settings(write_settings(tmp_path, text))
    assert "yolo99" in str(exc.value)


def test_species_must_map_to_a_known_category(tmp_path):
    text = real_settings().replace("    dolphin: marine_mammals", "    dolphin: sea_creatures")

    with pytest.raises(ValidationError) as exc:
        load_settings(write_settings(tmp_path, text))
    assert "sea_creatures" in str(exc.value)


def test_fusion_weights_must_add_up(tmp_path):
    text = real_settings().replace("siglip_weight: 0.3", "siglip_weight: 0.5")

    with pytest.raises(ValidationError) as exc:
        load_settings(write_settings(tmp_path, text))
    assert "add up to 1" in str(exc.value)
