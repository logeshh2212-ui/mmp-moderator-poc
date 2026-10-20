import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"
load_dotenv(PROJECT_ROOT / ".env")


class StrictModel(BaseModel):
    """Unknown keys are an error, so a typo in settings.yaml stops the app at startup."""

    model_config = ConfigDict(extra="forbid")


class LimitsSettings(StrictModel):
    image_max_bytes: int
    video_max_bytes: int
    max_video_duration_seconds: int
    duration_tolerance_seconds: float
    min_image_side_pixels: int
    image_max_pixels: int


class FormatsSettings(StrictModel):
    image: list[str]
    video: list[str]


class TimeoutsSettings(StrictModel):
    ffprobe_seconds: int
    frame_decode_seconds: int
    frame_sample_seconds: int


class RedisSettings(StrictModel):
    url: str
    cache_ttl_seconds: int
    lock_ttl_seconds: int
    lock_wait_seconds: float
    lock_poll_seconds: float
    connect_timeout_seconds: float
    socket_timeout_seconds: float


class IntakeSettings(StrictModel):
    chunk_size_bytes: int


class ToolsSettings(StrictModel):
    ffprobe: str
    ffmpeg: str


class MetadataSettings(StrictModel):
    ai_generator_keywords: list[str]


class LoggingSettings(StrictModel):
    level: str


class SafetySettings(StrictModel):
    # Only "placeholder" exists today. A typo here stops the app at startup.
    csam_provider: Literal["placeholder"]

class BlocklistSettings(StrictModel):
    max_distance: int = Field(ge=0, le=7)
    auto_block_codes: list[str]

class PrepareSettings(StrictModel):
    backbone_size: int = Field(gt=0)
    detector_size: int = Field(gt=0)
    short_video_seconds: float = Field(gt=0)
    short_video_interval_seconds: float = Field(gt=0)
    long_video_interval_seconds: float = Field(gt=0)
    last_frame_offset_seconds: float = Field(ge=0)
    max_video_frames: int = Field(ge=3)
    ffmpeg_hwaccel: str  # "none", "auto", "d3d11va", "qsv"

class ModelSettings(StrictModel):
    repo: str          # Hugging Face repo, e.g. google/siglip2-base-patch16-384
    revision: str      # exact commit hash, so the model can never change silently
    licence: str


class MLSettings(StrictModel):
    device: Literal["cpu", "cuda"]  # "openvino" is added in Step 16c
    dtype: Literal["float32", "float16"]
    num_threads: int = Field(ge=1)
    warmup: bool
    backbone: str | None
    detector: str | None
    species_model: str | None
    batch_size: int = Field(ge=1)
    models: dict[str, ModelSettings]
    prompts: dict[str, str]

    @model_validator(mode="after")
    def chosen_models_are_listed(self):
        for role in ("backbone", "detector", "species_model"):
            name = getattr(self, role)
            if name is not None and name not in self.models:
                raise ValueError(f"ml.{role} '{name}' is not listed in ml.models")
        return self

    @model_validator(mode="after")
    def float16_needs_a_gpu(self):
        if self.dtype == "float16" and self.device == "cpu":
            raise ValueError("ml.dtype float16 needs ml.device: cuda")
        return self



class TargetsSettings(StrictModel):
    image_p95_seconds: float
    video_p95_seconds: float


class Gate1Thresholds(StrictModel):
    pass_photo: float = Field(ge=0, le=100)
    pass_illustrated: float = Field(ge=0, le=100)
    pass_insect: float = Field(ge=0, le=100)
    fail_below: float = Field(ge=0, le=100)


class Gate1DetectorSettings(StrictModel):
    score_floor: float = Field(ge=0, le=1)
    score_full: float = Field(gt=0, le=1)
    min_box_score: float = Field(ge=0, le=1)
    max_species_boxes: int = Field(ge=1)
    lookalikes: list[str]
    video_frames: int = Field(ge=1)

    @model_validator(mode="after")
    def full_above_floor(self):
        if self.score_full <= self.score_floor:
            raise ValueError("gate1.detector.score_full must be above score_floor")
        return self


class Gate1FusionSettings(StrictModel):
    detector_weight: float = Field(ge=0, le=1)
    siglip_weight: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def weights_add_up_to_one(self):
        if abs(self.detector_weight + self.siglip_weight - 1) > 1e-6:
            raise ValueError("gate1.fusion weights must add up to 1")
        return self


class Gate1Settings(StrictModel):
    version: str
    enforce: bool
    approve_on_pass: bool
    thresholds: Gate1Thresholds
    video_min_pass_fraction: float = Field(gt=0, le=1)
    score_scale: float = Field(gt=0)
    templates: list[str]
    animals: dict[str, list[str]]
    insect_categories: list[str]
    wild_check_categories: list[str]
    always_review_categories: list[str]
    no_pet: list[str]
    photo: list[str]
    illustration: list[str]
    wild: list[str]
    domestic: list[str]
    detector: Gate1DetectorSettings
    fusion: Gate1FusionSettings
    species_min_confidence: float = Field(ge=0, le=1)
    species_category_min_confidence: float = Field(ge=0, le=1)
    species: dict[str, str]   # species name -> spec 2.1 category

    @model_validator(mode="after")
    def lists_are_consistent(self):
        categories = (
            self.insect_categories + self.wild_check_categories + self.always_review_categories
            + list(self.species.values())
        )
        unknown = set(categories) - set(self.animals)
        if unknown:
            raise ValueError(f"gate1 categories not in gate1.animals: {sorted(unknown)}")
        if any("{}" not in t for t in self.templates):
            raise ValueError("every gate1.templates entry needs a {} placeholder")
        return self



class AppSettings(StrictModel):
    limits: LimitsSettings
    allowed_formats: FormatsSettings
    timeouts: TimeoutsSettings
    temp_dir: Path
    redis: RedisSettings
    model_version: str
    intake: IntakeSettings
    tools: ToolsSettings
    metadata: MetadataSettings
    logging: LoggingSettings
    safety: SafetySettings
    blocklist: BlocklistSettings
    prepare: PrepareSettings
    ml: MLSettings
    gate1: Gate1Settings
    targets: TargetsSettings


class UniqueKeyLoader(yaml.SafeLoader):
    """YAML normally lets a later duplicate key silently replace the first one."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                line = key_node.start_mark.line + 1
                raise ValueError(f"Duplicate key '{key}' in settings file (line {line})")
            seen.add(key)
        return super().construct_mapping(node, deep)


def load_settings(path: Path = CONFIG_FILE) -> AppSettings:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.load(file, Loader=UniqueKeyLoader)

    # Environment variables can override the Redis URL.
    data["redis"]["url"] = os.getenv("REDIS_URL", data["redis"]["url"])
    # The cloud GPU run (Step 16b) switches device and precision without editing this file.
    data["ml"]["device"] = os.getenv("ML_DEVICE", data["ml"]["device"])
    data["ml"]["dtype"] = os.getenv("ML_DTYPE", data["ml"]["dtype"])


    return AppSettings.model_validate(data)


settings = load_settings()
