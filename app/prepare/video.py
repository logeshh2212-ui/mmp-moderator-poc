import logging
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from PIL import Image

from app.core.config import settings
from app.prepare.images import frame_from, to_rgb
from app.prepare.media import PreparedMedia
from app.validation.errors import ValidationFailure
from app.validation.video_check import run_tool


logger = logging.getLogger("mmp.prepare")

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
    """ffmpeg 'select': keep the first frame at or after each wanted time (a time of 0 means
    the first frame the decoder sees)."""
    terms = ["isnan(prev_t)"] if any(t <= 0 for t in times) else []
    terms += [f"gte(t,{t})*lt(prev_t,{t})" for t in times if t > 0]
    return "+".join(terms) or "0"


def fit(width: int, height: int, size: int) -> tuple[int, int]:
    """Largest even width x height inside size x size with the same shape (for scale_cuda)."""
    scale = min(size / max(width, 1), size / max(height, 1), 1.0)
    return max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)


def decode_piece(path: Path, start: float, length: float | None, times: list[float],
                 gpu_size: tuple[int, int] | None) -> list[tuple[bytes, float]]:
    """One ffmpeg run over [start, start + length): the wanted frames as (PNG, real time).
    gpu_size = decode and shrink on the NVIDIA GPU (NVDEC + scale_cuda) to that size."""
    p = settings.prepare
    wanted = select_expression([round(t - start, 3) for t in times])
    if gpu_size:
        w, h = gpu_size
        # showinfo after hwdownload: it reads the frame's real time from normal memory
        video_filter = f"select='{wanted}',scale_cuda={w}:{h},hwdownload,format=nv12,showinfo"
        hw = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    else:
        size = p.detector_size
        video_filter = (f"select='{wanted}',showinfo,"  # prints each kept frame's real time
                        f"scale=w={size}:h={size}:force_original_aspect_ratio=decrease")
        hw = ["-hwaccel", p.ffmpeg_hwaccel] if p.ffmpeg_hwaccel not in ("none", "cuda") else []
    seek = ["-ss", str(start)] if start > 0 else []
    span = ["-t", str(length)] if length is not None else []
    args = [settings.tools.ffmpeg, "-hide_banner", "-loglevel", "info", *hw, *seek, *span,
            "-i", str(path), "-map", "0:v:0", "-vf", video_filter, "-fps_mode", "passthrough",
            "-f", "image2pipe", "-c:v", "png", "-"]
    result = run_tool(args, settings.timeouts.frame_sample_seconds)
    pngs = [chunk + PNG_END for chunk in result.stdout.split(PNG_END) if chunk]
    if result.returncode != 0 or not pngs:
        raise ValidationFailure("corrupt_file")
    times_out = [start + float(t) for t in PTS_TIME.findall(result.stderr)]
    return [(png, times_out[i] if i < len(times_out) else None) for i, png in enumerate(pngs)]


def decode(path: Path, duration: float, times: list[float], gpu_size: tuple[int, int] | None,
           pieces: int) -> list[tuple[bytes, float]]:
    """The wanted frames, decoding `pieces` stretches of the video at the same time (the T4
    has two hardware decoders: 2 pieces = 3.5x faster than the CPU, measured 2026-10-28)."""
    pieces = max(1, min(pieces, len(times)))
    if pieces == 1:
        return decode_piece(path, 0.0, None, times, gpu_size)
    step = duration / pieces
    jobs = []
    for k in range(pieces):
        start = k * step
        end = None if k == pieces - 1 else (k + 1) * step
        own = [t for t in times if t >= start and (end is None or t < end)]
        if own:
            jobs.append((start, None if end is None else end - start, own))
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:  # threads only wait on ffmpeg
        parts = list(pool.map(lambda job: decode_piece(path, job[0], job[1], job[2], gpu_size), jobs))
    return [frame for part in parts for frame in part]


def prepare_video(path: Path, duration: float, width: int = 0, height: int = 0) -> PreparedMedia:
    """Decode the video; ffmpeg picks the frames, turns them upright and shrinks them.
    With ffmpeg_hwaccel cuda the NVIDIA GPU decodes and shrinks; any GPU failure (no GPU, a
    codec it cannot decode, a rotated phone video) falls back to the CPU, never to an error."""
    p = settings.prepare
    times = sample_timestamps(duration)
    if p.ffmpeg_hwaccel == "cuda" and width and height:
        try:
            frames = decode(path, duration, times, fit(width, height, p.detector_size), p.decode_pieces)
        except ValidationFailure:
            logger.warning("GPU video decode failed for %s; decoding on the CPU instead", path.name)
            frames = decode(path, duration, times, None, 1)
    else:
        frames = decode(path, duration, times, None, 1)  # CPU pieces gained nothing on 2 cores

    prepared = []
    for png, time in frames:
        with Image.open(BytesIO(png)) as img:
            rgb = to_rgb(img)
        prepared.append(frame_from(rgb, time))
    return PreparedMedia(frames=prepared)
