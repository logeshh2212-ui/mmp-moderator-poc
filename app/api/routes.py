from time import perf_counter
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, UploadFile

from app.api.schemas import ContentContext, ContentType, ModerationResponse
from app.core.config import settings
from app.validation.errors import ValidationFailure
from app.validation.intake import remove_temp_file, save_upload


router = APIRouter()


@router.post(
    "/api/v1/moderate",
    response_model=ModerationResponse,
)
async def moderate_content(
    file: Annotated[UploadFile, File()],
    content_type: Annotated[ContentType, Form()],
    user_id: Annotated[UUID, Form()],
    context: Annotated[ContentContext, Form()] = "post",
) -> ModerationResponse:
    start = perf_counter()
    content_id = uuid4()
    intake = None

    # user_id is used for Redis in Step 10; context for gates in later phases.
    _ = user_id, context

    try:
        intake = await save_upload(file, content_type, content_id)
        # Steps 5-8 will run here. The orchestrator (Step 9) replaces this block.
        decision, rejection_code, flags = (
            "HUMAN_REVIEW", None, ["validation_not_implemented"]
        )
    except ValidationFailure as failure:
        decision, rejection_code, flags = "REJECTED", "MOD_008", [failure.flag]
    finally:
        if intake is not None:
            remove_temp_file(intake.path)

    return ModerationResponse(
        content_id=content_id,
        decision=decision,
        hard_reject=False,
        rejection_code=rejection_code,
        flags=flags,
        processing_ms=int((perf_counter() - start) * 1000),
        model_version=settings.model_version,
    )
