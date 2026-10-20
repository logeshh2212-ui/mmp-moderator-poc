from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from uuid import UUID

from app.api.schemas import Decision, DimensionScores, ModerationResponse
from app.core.config import settings
from app.validation.result import ValidationResult
from app.validation.csam import CsamResult
from app.cache.blocklist import BlocklistMatch



@dataclass
class ModerationContext:
    """One object per upload. Each pipeline step reads what it needs and adds its results."""

    # From the request
    content_id: UUID
    user_id: UUID
    content_type: str
    context: str
    started_at: float

    # Step 4: intake
    path: Path | None = None
    size_bytes: int | None = None
    sha256: str | None = None

    # Steps 5-8: validation
    validation: ValidationResult | None = None
        # Step 8b: CSAM hash check
    csam: CsamResult | None = None

        # Step 10b: pHash blocklist
    blocklist_match: BlocklistMatch | None = None


    # Filled in by later phases (Gate 1, Gate 2, scoring, AI label)
    pet_score: float | None = None
    composite_score: float | None = None
    dimension_scores: DimensionScores | None = None
    species_detected: list[str] = field(default_factory=list)
    ai_generated: bool | None = None
    ai_score: float | None = None

    # The decision so far
    decision: Decision = "HUMAN_REVIEW"
    rejection_code: str | None = None
    hard_reject: bool = False
    flags: list[str] = field(default_factory=list)

    def reject(self, code: str, flag: str, *, hard: bool = False) -> None:
        self.decision = "REJECTED"
        self.rejection_code = code
        self.hard_reject = hard
        self.flags.append(flag)

    def to_response(self) -> ModerationResponse:
        return ModerationResponse(
            content_id=self.content_id,
            decision=self.decision,
            pet_score=self.pet_score,
            composite_score=self.composite_score,
            ai_generated=self.ai_generated,
            ai_score=self.ai_score,
            dimension_scores=self.dimension_scores,
            hard_reject=self.hard_reject,
            rejection_code=self.rejection_code,
            flags=list(self.flags),
            species_detected=list(self.species_detected),
            processing_ms=int((perf_counter() - self.started_at) * 1000),
            model_version=settings.model_version,
        )
