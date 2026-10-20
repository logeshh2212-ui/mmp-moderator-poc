
from time import perf_counter
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, UploadFile

from app.api.schemas import ModerationResponse
from app.core.config import settings


router = APIRouter()


@router.post(
    "/api/v1/moderate",
    response_model=ModerationResponse,
)
async def moderate_content(
    file: Annotated[UploadFile, File()],
    content_type: Annotated[Literal["image", "video"], Form()],
    user_id: Annotated[UUID, Form()],
    context: Annotated[
        Literal["post", "profile", "breeding", "story"],
        Form(),
    ] = "post",
) -> ModerationResponse:
    start = perf_counter()

    # Temporary stub: real validation comes in later steps.
    _ = file, content_type, user_id, context

    return ModerationResponse(
        content_id=uuid4(),
        decision="HUMAN_REVIEW",
        hard_reject=False,
        flags=["validation_not_implemented"],
        processing_ms=int((perf_counter() - start) * 1000),
        model_version=settings.model_version,
    )
