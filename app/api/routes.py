from pathlib import Path
from time import perf_counter
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.schemas import ContentContext, ContentType, ModerationResponse
from app.cache.idempotency import IdempotencyStore, get_idempotency_store
from app.core.config import settings
from app.validation.errors import ValidationFailure
from app.validation.intake import remove_temp_file, save_upload


router = APIRouter()


def run_checks(path: Path, content_type: str) -> None:
    """Steps 5-8 add their checks here. Raise ValidationFailure to reject."""


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

    # context is used by the gates in later phases.
    _ = context

    try:
        intake = await save_upload(file, content_type, content_id)
    except ValidationFailure as failure:
        # We stopped reading, so there is no full hash: this answer is not cached.
        return build_response(content_id, start, failure)

    try:
        cached = await store.get_cached(intake.sha256, user_id)
        if cached is not None:
            return cached

        if not await store.lock(intake.sha256, user_id):
            cached = await store.wait_for_result(intake.sha256, user_id)
            if cached is not None:
                return cached
            raise HTTPException(
                status_code=409,
                detail="This upload is already being processed. Try again shortly.",
            )

        try:
            try:
                run_checks(intake.path, content_type)
                response = build_response(content_id, start)
            except ValidationFailure as failure:
                response = build_response(content_id, start, failure)

            # Only final decisions reach this line; errors skip it.
            await store.store(intake.sha256, user_id, response)
            return response
        finally:
            await store.release(intake.sha256, user_id)
    finally:
        remove_temp_file(intake.path)
