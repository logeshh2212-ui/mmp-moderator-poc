from pathlib import Path
from time import perf_counter
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import ContentContext, ContentType, ModerationResponse
from app.cache.idempotency import IdempotencyStore, UploadKey, get_idempotency_store
from app.core.config import settings
from app.validation.errors import ProcessingTimeout, ValidationFailure
from app.validation.formats import check_format
from app.validation.hashing import phash_image, phash_png_bytes
from app.validation.image_check import check_image
from app.validation.intake import remove_temp_file, save_upload
from app.validation.metadata import read_image_metadata, read_video_metadata
from app.validation.result import ValidationResult
from app.validation.video_check import check_video




router = APIRouter()


def run_checks(path: Path, content_type: str) -> ValidationResult:
    """Run every validation check. Raise ValidationFailure to reject."""
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




def build_response(
    content_id: UUID,
    start: float,
    failure: ValidationFailure | None = None,
) -> ModerationResponse:
    if failure is None:
        decision, rejection_code, flags = "HUMAN_REVIEW", None, ["validation_not_implemented"]
    else:
        decision, rejection_code, flags = "REJECTED", "MOD_008", [failure.flag]

    return ModerationResponse(
        content_id=content_id,
        decision=decision,
        hard_reject=False,
        rejection_code=rejection_code,
        flags=flags,
        processing_ms=int((perf_counter() - start) * 1000),
        model_version=settings.model_version,
    )


@router.post(
    "/api/v1/moderate",
    response_model=ModerationResponse,
)
async def moderate_content(
    file: Annotated[UploadFile, File()],
    content_type: Annotated[ContentType, Form()],
    user_id: Annotated[UUID, Form()],
    store: Annotated[IdempotencyStore, Depends(get_idempotency_store)],
    context: Annotated[ContentContext, Form()] = "post",
) -> ModerationResponse:
    start = perf_counter()
    content_id = uuid4()


    try:
        intake = await save_upload(file, content_type, content_id)
    except ValidationFailure as failure:
        # We stopped reading, so there is no full hash: this answer is not cached.
        return build_response(content_id, start, failure)

    try:
        validation = await run_in_threadpool(run_checks, intake.path, content_type)
        _ = validation  # Step 9 (orchestrator) puts this into the context object

        response = build_response(content_id, start)
    except ValidationFailure as failure:
        response = build_response(content_id, start, failure)
    except ProcessingTimeout:
        # Not a decision: skip storing, release the lock, let the caller retry.
        raise HTTPException(
                status_code=503,
                detail="Processing timed out. Try again shortly.",
            )


    key = UploadKey(intake.sha256, user_id, content_type, context)

    try:
        cached = await store.get_cached(key)
        if cached is not None:
            return cached

        if not await store.lock(key):
            cached = await store.wait_for_result(key)
            if cached is not None:
                return cached
            raise HTTPException(
                status_code=409,
                detail="This upload is already being processed. Try again shortly.",
            )

        try:
            try:
                await run_in_threadpool(run_checks, intake.path, content_type)
                response = build_response(content_id, start)
            except ValidationFailure as failure:
                response = build_response(content_id, start, failure)

            # Only final decisions reach this line; errors skip it.
            await store.store(key, response)
            return response
        finally:
            await store.release(key)
    finally:
        remove_temp_file(intake.path)


