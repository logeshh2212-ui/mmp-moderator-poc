import os
from pathlib import Path
from typing import Annotated, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.api.schemas import DimensionScores, RejectionCode


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"
# Files in model repos we never download or need: duplicate .bin / TensorFlow / Flax / ONNX
# copies of the .safetensors weights, and leftover training checkpoints.
MODEL_IGNORE_PATTERNS = ["*.bin", "*.pt", "*.h5", "*.msgpack", "*.onnx", "checkpoint-*/*"]
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
    # placeholder = no real check (POC); hash_list = exact SHA-256 / MD5 matches against a
    # hash file from an official source. PhotoDNA / Thorn Safer are added once there is access.
    csam_provider: Literal["placeholder", "hash_list"]
    require_real_check: bool        # production: refuse to start with the placeholder
    hash_list_file: Path | None     # hash_list: one hex hash per line, kept OUTSIDE the repo
    quarantine_dir: Path            # matched files and their report records (restricted access)

    @model_validator(mode="after")
    def provider_is_usable(self):
        if self.require_real_check and self.csam_provider == "placeholder":
            raise ValueError("safety.require_real_check is on but csam_provider is the placeholder")
        if self.csam_provider == "hash_list" and self.hash_list_file is None:
            raise ValueError("safety.csam_provider hash_list needs safety.hash_list_file")
        return self

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
    nsfw_model: str | None
    ai_model: str | None
    ocr: bool                       # read text on images / video frames (gate2.text.ocr)
    audio_model: str | None         # video sound events (gate2.audio)
    speech_model: str | None        # video speech to text (gate2.audio)
    batch_size: int = Field(ge=1)
    models: dict[str, ModelSettings]
    prompts: dict[str, str]

    @model_validator(mode="after")
    def chosen_models_are_listed(self):
        for role in ("backbone", "detector", "species_model", "nsfw_model", "ai_model", "audio_model", "speech_model"):
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



DIMENSIONS = tuple(DimensionScores.model_fields)   # D1_welfare ... D6_relevance
SafetyDimension = Literal["D1_welfare", "D2_nsfw", "D3_violence", "D4_prohibited"]
Dimension = Literal[
    "D1_welfare", "D2_nsfw", "D3_violence", "D4_prohibited", "D5_quality", "D6_relevance",
]


class Gate2Thresholds(StrictModel):
    approve: float = Field(ge=0, le=100)
    review: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def review_below_approve(self):
        if self.review >= self.approve:
            raise ValueError("gate2.thresholds.review must be below approve")
        return self


class Gate2Signal(StrictModel):
    dimension: SafetyDimension
    deduction: float = Field(ge=0, le=100)
    code: RejectionCode
    hard: bool = False
    review: bool = False
    # points off other dimensions too (spec 4.4: drug use on an animal also hits D1)
    also_deduct: dict[SafetyDimension, Annotated[float, Field(ge=0, le=100)]] = Field(
        default_factory=dict
    )


class Gate2Context(StrictModel):
    deductions: dict[str, float] = Field(default_factory=dict)   # signal -> replacement deduction
    dimension_factor: dict[SafetyDimension, float] = Field(default_factory=dict)
    review_on: list[str] = Field(default_factory=list)


Band = tuple[float, float]   # (lowest, highest) D5 score, spec 4.5


class Gate2QualityBands(StrictModel):
    sharp: Band
    slight_blur: Band
    heavy_blur: Band
    dark: Band
    screenshot: Band
    compressed: Band


class Gate2SharpnessSettings(StrictModel):
    heavy_below: float = Field(gt=0)
    slight_below: float = Field(gt=0)
    full_at: float = Field(gt=0)

    @model_validator(mode="after")
    def increasing(self):
        if not self.heavy_below < self.slight_below < self.full_at:
            raise ValueError("gate2.quality.sharpness needs heavy_below < slight_below < full_at")
        return self


class Gate2QualitySettings(StrictModel):
    bands: Gate2QualityBands
    blank_max_contrast: float = Field(ge=0)
    sharpness: Gate2SharpnessSettings
    min_box_side: int = Field(ge=3)
    dark_mean_below: float = Field(gt=0, le=255)
    bright_level: float = Field(gt=0, le=255)
    bright_fraction_above: float = Field(gt=0, lt=1)
    bar_level: float = Field(ge=0, le=255)
    bar_max_std: float = Field(ge=0)
    bar_min_fraction: float = Field(gt=0, lt=1)
    lowres_short_side_below: int = Field(gt=0)
    jpeg_quality_below: float = Field(gt=0, le=100)


class Gate2ItemsSettings(StrictModel):
    min_score: float = Field(gt=0, le=1)
    queries: dict[str, str | None] = Field(min_length=1)   # OWLv2 query -> signal (None = harmless)


class Gate2RelevanceKind(StrictModel):
    band: Band                                  # spec 4.6 D6 range for this kind of post
    sentences: list[str] = Field(min_length=1)  # what SigLIP compares the picture with


class Gate2RelevanceSettings(StrictModel):
    kinds: dict[str, Gate2RelevanceKind] = Field(min_length=2)


class OcrSettings(StrictModel):
    languages: list[str] = Field(min_length=1)   # EasyOCR codes, e.g. en, hi
    model_dir: Path                              # EasyOCR weights (downloaded by scripts.download_models)
    min_confidence: float = Field(ge=0, le=1)    # text boxes below this are ignored
    video_frames: int = Field(ge=1)              # detector frames of a video that are read


class Gate2TextSettings(StrictModel):
    signal: str                   # gate2 signal for a listed word in on-screen text or speech
    blocklists: list[Path] = Field(min_length=1)   # relative to the project folder
    allow: list[Path]             # words that are normal on a pet platform (bitch, cock...)
    ocr: OcrSettings

    @field_validator("blocklists", "allow")
    @classmethod
    def files_exist(cls, paths):
        resolved = [p if p.is_absolute() else PROJECT_ROOT / p for p in paths]
        missing = [str(p) for p in resolved if not p.is_file()]
        if missing:
            raise ValueError(f"word list files not found: {missing}")
        return resolved


class Gate2SoundGroup(StrictModel):
    labels: list[str] = Field(min_length=1)    # AudioSet label names, exactly as the model has them
    min_score: float = Field(gt=0, lt=1)


class Gate2AudioSettings(StrictModel):
    sample_rate: int = Field(gt=0)
    chunk_seconds: float = Field(gt=0)
    sounds: dict[str, Gate2SoundGroup]          # gate2 signal -> the sounds that raise it
    speech_signal: str                          # a listed word in the speech
    speech_label: str                           # AudioSet label that means someone is talking
    speech_min_score: float = Field(gt=0, lt=1) # Whisper runs only from this score (or with no sound model)


class Gate2HarmIdea(StrictModel):
    signals: list[str] = Field(min_length=1)    # gate2.signals this scene stands for
    sentences: list[str] = Field(min_length=1)


class Gate2SceneSettings(StrictModel):
    min_probability: float = Field(gt=0, lt=1)           # SigLIP yes-probability for a harm sentence
    context_min_probability: float = Field(gt=0, lt=1)   # same for a context sentence
    safe: dict[str, list[str]] = Field(min_length=1)     # ordinary scenes a harm sentence must beat
    harm: dict[str, Gate2HarmIdea] = Field(min_length=1)
    contexts: dict[str, list[str]]                       # gate2.contexts name -> sentences


class Gate2NsfwSettings(StrictModel):
    label: str                       # the model's label for nudity
    # D2 signal -> probability from which it counts; the highest level reached wins
    levels: dict[str, Annotated[float, Field(gt=0, lt=1)]] = Field(min_length=1)


class Gate2Settings(StrictModel):
    version: str
    enforce: bool
    flag_zero_shot: bool
    weights: dict[Dimension, float]
    thresholds: Gate2Thresholds
    review_from_deduction: float = Field(gt=0, le=100)
    video_window_frames: int = Field(ge=1)
    code_priority: list[RejectionCode]
    low_score_code: RejectionCode
    signals: dict[str, Gate2Signal]
    contexts: dict[str, Gate2Context]
    quality: Gate2QualitySettings
    items: Gate2ItemsSettings
    relevance: Gate2RelevanceSettings
    scene: Gate2SceneSettings
    nsfw: Gate2NsfwSettings
    text: Gate2TextSettings
    audio: Gate2AudioSettings

    @model_validator(mode="after")
    def policy_is_consistent(self):
        if set(self.weights) != set(DIMENSIONS):
            raise ValueError(f"gate2.weights needs exactly {list(DIMENSIONS)}")
        if abs(sum(self.weights.values()) - 1) > 1e-6:
            raise ValueError("gate2.weights must add up to 1")
        if sorted(self.code_priority) != sorted(RejectionCode.__args__):
            raise ValueError("gate2.code_priority must list every rejection code once")
        for name, context in self.contexts.items():
            named = set(context.deductions) | set(context.review_on)
            unknown = named - set(self.signals)
            if unknown:
                raise ValueError(f"gate2.contexts.{name} names unknown signals: {sorted(unknown)}")
            hard = sorted(s for s in named if self.signals[s].hard)
            if hard:
                raise ValueError(f"gate2.contexts.{name} cannot change hard-reject signals: {hard}")
        for query, signal in self.items.queries.items():
            if signal is not None and (
                signal not in self.signals or self.signals[signal].dimension != "D4_prohibited"
            ):
                raise ValueError(f"gate2.items.queries.{query}: {signal} is not a D4 signal")
        for idea, harm in self.scene.harm.items():
            unknown = set(harm.signals) - set(self.signals)
            if unknown:
                raise ValueError(f"gate2.scene.harm.{idea} names unknown signals: {sorted(unknown)}")
        unknown = set(self.scene.contexts) - set(self.contexts)
        if unknown:
            raise ValueError(f"gate2.scene.contexts names unknown contexts: {sorted(unknown)}")
        if self.text.signal not in self.signals:
            raise ValueError(f"gate2.text.signal: unknown signal {self.text.signal}")
        for signal in [*self.audio.sounds, self.audio.speech_signal]:
            if signal not in self.signals or self.signals[signal].hard:
                raise ValueError(f"gate2.audio: {signal} must be a known, non-hard signal (spec 6.2)")
        for signal in self.nsfw.levels:
            if signal not in self.signals or self.signals[signal].dimension != "D2_nsfw":
                raise ValueError(f"gate2.nsfw.levels: {signal} is not a D2 signal")
        return self


class AiLabelSettings(StrictModel):
    version: str
    threshold: float = Field(gt=0, lt=1)   # model probability from which ai_generated is true
    crops: Literal[1, 3]                   # Smaug: 3 crops = most accurate, 1 = 3x faster
    video_frames: int = Field(ge=1)        # detector frames of a video the model looks at


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
    gate2: Gate2Settings
    ai_label: AiLabelSettings
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
