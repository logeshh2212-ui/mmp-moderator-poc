from pathlib import Path

import numpy as np

from app.core.config import settings
from app.validation.video_check import run_tool


def load_audio(path: Path) -> np.ndarray | None:
    """The first audio track as mono float32 samples at gate2.audio.sample_rate, straight from
    ffmpeg's stdout (no temp file). None if it cannot be decoded: the video is then judged
    on its pictures alone."""
    rate = settings.gate2.audio.sample_rate
    result = run_tool(
        [settings.tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-map", "0:a:0", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"],
        settings.timeouts.frame_sample_seconds,
    )
    if result.returncode != 0 or len(result.stdout) < 2:
        return None
    samples = np.frombuffer(result.stdout[: len(result.stdout) // 2 * 2], dtype=np.int16)
    return samples.astype(np.float32) / 32768.0
