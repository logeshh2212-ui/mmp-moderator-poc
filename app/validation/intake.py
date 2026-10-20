import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile

from app.core.config import settings
from app.validation.errors import ValidationFailure


@dataclass
class IntakeResult:
    path: Path
    size_bytes: int
    sha256: str


def max_bytes_for(content_type: str) -> int:
    if content_type == "video":
        return settings.limits.video_max_bytes
    return settings.limits.image_max_bytes


async def save_upload(
    file: UploadFile,
    content_type: str,
    content_id: UUID,
) -> IntakeResult:
    """Stream the upload to disk in chunks, hashing and counting as we go."""
    temp_dir = settings.temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"{content_id}.upload"

    limit = max_bytes_for(content_type)
    chunk_size = settings.intake.chunk_size_bytes
    digest = hashlib.sha256()
    size = 0

    try:
        with path.open("wb") as out:
            while chunk := await file.read(chunk_size):
                size += len(chunk)
                if size > limit:
                    raise ValidationFailure("file_too_large")
                digest.update(chunk)
                out.write(chunk)

        if size == 0:
            raise ValidationFailure("empty_file")
    except BaseException:
        remove_temp_file(path)
        raise

    return IntakeResult(path=path, size_bytes=size, sha256=digest.hexdigest())


def remove_temp_file(path: Path) -> None:
    path.unlink(missing_ok=True)
