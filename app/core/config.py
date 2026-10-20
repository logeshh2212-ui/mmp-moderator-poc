import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


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

    return AppSettings.model_validate(data)


settings = load_settings()
