import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from app.core.config import settings
from app.validation.errors import ProcessingTimeout, ValidationFailure


# ffprobe's format_name for each container we accept. mp4 and mov share one
# demuxer ("mov,mp4,m4a,3gp,3g2,mj2"), so both names appear for either file.
CONTAINER_NAMES = {"mp4": "mp4", "mov": "mov", "avi": "avi"}


@dataclass
class VideoInfo:
    duration_seconds: float
    width: int
    height: int
    fps: float | None
    video_codec: str
    has_audio: bool
    audio_codec: str | None
    first_frame_png: bytes  # Step 8 computes the perceptual hash from this


def run_tool(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise ProcessingTimeout(f"{args[0]} timed out after {timeout}s")


def probe(path: Path) -> dict:
    result = run_tool(
        [
            settings.tools.ffprobe,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        settings.timeouts.ffprobe_seconds,
    )
    if result.returncode != 0:
        raise ValidationFailure("corrupt_file")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise ValidationFailure("corrupt_file")


def parse_fps(rate: str | None) -> float | None:
    """ffprobe gives rates like '30000/1001'; '0/0' means unknown."""
    try:
        value = Fraction(rate)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return round(float(value), 3) if value > 0 else None


def parse_duration(info: dict, video_stream: dict) -> float | None:
    for source in (info.get("format", {}), video_stream):
        try:
            return float(source["duration"])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def decode_first_frame(path: Path) -> bytes:
    """Really decode one frame; a file that cannot do this is corrupt."""
    result = run_tool(
        [
            settings.tools.ffmpeg,
            "-v", "error",
            "-i", str(path),
            "-map", "0:v:0",
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-",
        ],
        settings.timeouts.frame_decode_seconds,
    )
    if result.returncode != 0 or not result.stdout:
        raise ValidationFailure("corrupt_file")
    return result.stdout


def check_video(path: Path, file_format: str) -> VideoInfo:
    """Probe and test-decode the video. Raise ValidationFailure if it is bad."""
    info = probe(path)

    format_names = info.get("format", {}).get("format_name", "").split(",")
    if CONTAINER_NAMES.get(file_format) not in format_names:
        raise ValidationFailure("type_mismatch")

    streams = info.get("streams", [])
    video_streams = [
        s for s in streams
        if s.get("codec_type") == "video"
        # Cover art in audio files shows up as a one-frame "video" stream.
        and not s.get("disposition", {}).get("attached_pic")
    ]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not video_streams:
        raise ValidationFailure("no_video_stream")
    video = video_streams[0]

    duration = parse_duration(info, video)
    if duration is None:
        raise ValidationFailure("corrupt_file")

    limits = settings.limits
    if duration > limits.max_video_duration_seconds + limits.duration_tolerance_seconds:
        raise ValidationFailure("duration_exceeded")

    first_frame = decode_first_frame(path)
    audio = audio_streams[0] if audio_streams else None

    return VideoInfo(
        duration_seconds=round(duration, 3),
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        fps=parse_fps(video.get("avg_frame_rate")),
        video_codec=video.get("codec_name", "unknown"),
        has_audio=audio is not None,
        audio_codec=audio.get("codec_name") if audio else None,
        first_frame_png=first_frame,
    )
