
import os
from pathlib import Path

import yaml
from pydantic import BaseModel
from dotenv import load_dotenv




PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"
load_dotenv(PROJECT_ROOT / ".env")


class LimitsSettings(BaseModel):
    image_max_bytes: int
    video_max_bytes: int
    max_video_duration_seconds: int
    duration_tolerance_seconds: float
    min_image_side_pixels: int


class FormatsSettings(BaseModel):
    image: list[str]
    video: list[str]


class TimeoutsSettings(BaseModel):
    ffprobe_seconds: int
    frame_decode_seconds: int


class RedisSettings(BaseModel):
    url: str
    cache_ttl_seconds: int
    lock_ttl_seconds: int

class IntakeSettings(BaseModel):
    chunk_size_bytes: int


class AppSettings(BaseModel):
    limits: LimitsSettings
    allowed_formats: FormatsSettings
    timeouts: TimeoutsSettings
    temp_dir: Path
    redis: RedisSettings
    model_version: str
    intake:IntakeSettings




def load_settings() -> AppSettings:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    # Environment variables can override the Redis URL.
    data["redis"]["url"] = os.getenv(
        "REDIS_URL",
        data["redis"]["url"],
    )

    return AppSettings.model_validate(data)


settings = load_settings()
