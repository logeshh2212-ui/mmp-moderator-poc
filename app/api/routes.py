from time import perf_counter
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from redis.exceptions import RedisError

from app.api.schemas import ContentContext, ContentType, ModerationResponse
from app.cache.idempotency import IdempotencyStore, UploadKey, get_idempotency_store
from app.core.logging import log_problem, log_result
from app.core.versioning import current_model_version
from app.pipeline.context import ModerationContext
from app.pipeline.orchestrator import finish, run_pipeline
from app.validation.errors import ProcessingTimeout, ValidationFailure
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
    store: Annotated[IdempotencyStore, Depends(get_idempotency_store)],
    context: Annotated[ContentContext, Form()] = "post",
) -> ModerationResponse:
    ctx = ModerationContext(
        content_id=uuid4(),
        user_id=user_id,
        content_type=content_type,
        context=context,
        started_at=perf_counter(),
    )

    # Step 4 runs here, not in the orchestrator: it needs the upload stream, and
    # its SHA-256 is the idempotency key we must check before any heavy work.
    try:
        intake = await save_upload(file, content_type, ctx.content_id)
    except ValidationFailure as failure:
        # We stopped reading, so there is no full hash: this answer is not cached.
        ctx.reject("MOD_008", failure.flag)
        return finish(ctx)

    ctx.path = intake.path
    ctx.size_bytes = intake.size_bytes
    ctx.sha256 = intake.sha256
    key = UploadKey(intake.sha256, user_id, content_type, context, current_model_version())

    try:
        cached = await store.get_cached(key)
        if cached is not None:
            log_result(cached, cached=True)
            return cached

        if not await store.lock(key):
            cached = await store.wait_for_result(key)
            if cached is not None:
                log_result(cached, cached=True)
                return cached
            log_problem(409, ctx.content_id, "same upload is already being processed")
            raise HTTPException(
                status_code=409,
                detail="This upload is already being processed. Try again shortly.",
            )

        try:
            try:
                response = await run_in_threadpool(run_pipeline, ctx)
            except ProcessingTimeout as exc:
                # Not a decision: skip storing, release the lock, let the caller retry.
                log_problem(503, ctx.content_id, str(exc))
                raise HTTPException(
                    status_code=503,
                    detail="Processing timed out. Try again shortly.",
                )

            # Only final decisions reach this line; errors skip it.
            await store.store(key, response)
            return response
        finally:
            await store.release(key)
    except RedisError as exc:
        # Redis is down: no duplicate protection and no blocklist, so no decision.
        log_problem(503, ctx.content_id, f"redis unavailable: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Duplicate-protection store unavailable. Try again shortly.",
        )
    finally:
        remove_temp_file(intake.path)
