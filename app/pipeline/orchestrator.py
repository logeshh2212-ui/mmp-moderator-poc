from pathlib import Path

from app.api.schemas import ModerationResponse
from app.core.logging import log_result
from app.pipeline.context import ModerationContext
from app.validation.errors import ValidationFailure
from app.validation.formats import check_format
from app.validation.hashing import phash_image, phash_png_bytes
from app.validation.image_check import check_image
from app.validation.metadata import read_image_metadata, read_video_metadata
from app.validation.result import ValidationResult
from app.validation.video_check import check_video
from app.validation.csam import get_csam_checker


def run_checks(path: Path, content_type: str) -> ValidationResult:
    """Steps 5-8. Raise ValidationFailure to reject."""
    file_format = check_format(path, content_type)

    if content_type == "image":
        image = check_image(path)
        return ValidationResult(
            file_format=file_format,
            image=image,
            metadata=read_image_metadata(path),
            phash=phash_image(path, image.orientation),
        )

    video = check_video(path, file_format)
    return ValidationResult(
        file_format=file_format,
        video=video,
        metadata=read_video_metadata(video.tags),
        phash=phash_png_bytes(video.first_frame_png),
    )


def run_pipeline(ctx: ModerationContext) -> ModerationResponse:
    """Run every step after intake, in order, stopping at the first failure.

    ProcessingTimeout is deliberately not caught: a timeout is not a decision.
    """
    try:
        ctx.validation = run_checks(ctx.path, ctx.content_type)
    except ValidationFailure as failure:
        ctx.reject("MOD_008", failure.flag)
        return finish(ctx)

    # Step 8b: must run before any model or reviewer could see the file.
    ctx.csam = get_csam_checker().check(ctx.path, ctx.sha256)
    if ctx.csam.match:
        # A real provider will also move the file to restricted quarantine
        # storage and start legal reporting before the temp file is deleted.
        ctx.reject("MOD_004", "quarantined", hard=True)
        return finish(ctx)

    ctx.flags.append("validation_passed")

    # Later phases each add one call here, for example:
    #   prepare(ctx)        Steps 13-14: resize, sample frames
    #   gate1(ctx)          Phase 3: pet score; may reject with MOD_001
    #   if ctx.decision == "REJECTED":
    #       return finish(ctx)
    return finish(ctx)



def finish(ctx: ModerationContext) -> ModerationResponse:
    response = ctx.to_response()
    log_result(response)
    return response
