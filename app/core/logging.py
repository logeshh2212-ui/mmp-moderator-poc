import logging
from uuid import UUID

from app.api.schemas import ModerationResponse
from app.core.config import settings


logger = logging.getLogger("mmp.moderation")


def setup_logging() -> None:
    logging.basicConfig(
        level=settings.logging.level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def log_result(response: ModerationResponse, *, cached: bool = False) -> None:
    """One line per moderated request: id, decision, reason, time."""
    reason = "-"
    if response.rejection_code:
        reason = f"{response.rejection_code}:{','.join(response.flags)}"
    logger.info(
        "content_id=%s decision=%s reason=%s ms=%d cached=%s",
        response.content_id,
        response.decision,
        reason,
        response.processing_ms,
        cached,
    )


def log_problem(status: int, content_id: UUID, detail: str) -> None:
    """One line for requests that end without a decision (409, 503)."""
    logger.warning("content_id=%s status=%d detail=%s", content_id, status, detail)
