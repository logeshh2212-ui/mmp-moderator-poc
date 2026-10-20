from dataclasses import dataclass

from app.validation.image_check import ImageInfo
from app.validation.metadata import MediaMetadata
from app.validation.video_check import VideoInfo


@dataclass
class ValidationResult:
    """Everything validation learned about one upload. Step 9 moves this into the pipeline context."""

    file_format: str
    metadata: MediaMetadata
    phash: str | None
    image: ImageInfo | None = None
    video: VideoInfo | None = None
