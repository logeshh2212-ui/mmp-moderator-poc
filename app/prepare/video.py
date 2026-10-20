import re
from io import BytesIO
from pathlib import Path

from PIL import Image

from app.core.config import settings
from app.prepare.images import frame_from, to_rgb
from app.prepare.media import PreparedMedia
from app.validation.errors import ValidationFailure
from app.validation.video_check import run_tool


PNG_END = b"IEND\xaeB`\x82"                  # last 8 bytes of every PNG file
PTS_TIME = re.compile(rb"pts_time:\s*([0-9.]+)")


def sample_timestamps(duration: float) -> list[float]:
    """Graph Step 2: every 0.5 s under 10 s, else 1 fps; plus first, middle and last."""
    p = settings.prepare
    step = p.short_video_interval_seconds if duration < p.short_video_seconds else p.long_video_interval_seconds

    times = {round(i * step, 3) for i in range(int(duration / step) + 1) if i * step < duration}
    times.add(round(duration / 2, 3))
    times.add(round(max(0.0, duration - p.last_frame_offset_seconds), 3))
    ordered = sorted(times)

    limit = p.max_video_frames
    if len(ordered) > limit:
        # Thin out evenly; the first and last frame are always kept.
        picks = {round(i * (len(ordered) - 1) / (limit - 1)) for i in range(limit)}
        ordered = [ordered[i] for i in sorted(picks)]
    return ordered


def select_expression(times: list[float]) -> str:
    """ffmpeg 'select': keep the first frame at or after each wanted time."""
    terms = ["isnan(prev_t)"]  # the very first frame
    terms += [f"gte(t,{t})*lt(prev_t,{t})" for t in times if t > 0]
    return "+".join(terms)


def prepare_video(path: Path, duration: float) -> PreparedMedia:
    """Decode the video once; ffmpeg picks the frames, turns them upright and shrinks them."""
    p = settings.prepare
    size = p.detector_size
    video_filter = (
        f"select='{select_expression(sample_timestamps(duration))}',"
        "showinfo,"  # prints each kept frame's real time to stderr
        f"scale=w={size}:h={size}:force_original_aspect_ratio=decrease"
    )

    args = [settings.tools.ffmpeg, "-hide_banner", "-loglevel", "info"]
    if p.ffmpeg_hwaccel != "none":
        args += ["-hwaccel", p.ffmpeg_hwaccel]
    args += [
        "-i", str(path),
        "-map", "0:v:0",
        "-vf", video_filter,
        "-fps_mode", "passthrough",
        "-f", "image2pipe",
        "-c:v", "png",
        "-",
    ]
    result = run_tool(args, settings.timeouts.frame_sample_seconds)

    pngs = [chunk + PNG_END for chunk in result.stdout.split(PNG_END) if chunk]
    if result.returncode != 0 or not pngs:
        raise ValidationFailure("corrupt_file")
    times = [float(t) for t in PTS_TIME.findall(result.stderr)]

    frames = []
    for i, png in enumerate(pngs):
        with Image.open(BytesIO(png)) as img:
            rgb = to_rgb(img)
        frames.append(frame_from(rgb, times[i] if i < len(times) else None))
    return PreparedMedia(frames=frames)
