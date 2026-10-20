from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


ContentType = Literal["image", "video"]
ContentContext = Literal["post", "profile", "breeding", "story"]
Decision = Literal["APPROVED", "REJECTED", "HUMAN_REVIEW"]
RejectionCode = Literal[
    "MOD_001", "MOD_002", "MOD_003", "MOD_004", "MOD_005",
    "MOD_006", "MOD_007", "MOD_008", "MOD_009",
]

Score = Field(default=None, ge=0, le=100)


class DimensionScores(BaseModel):
    D1_welfare: float = Field(ge=0, le=100)
    D2_nsfw: float = Field(ge=0, le=100)
    D3_violence: float = Field(ge=0, le=100)
    D4_prohibited: float = Field(ge=0, le=100)
    D5_quality: float = Field(ge=0, le=100)
    D6_relevance: float = Field(ge=0, le=100)


class ModerationResponse(BaseModel):
    content_id: UUID
    decision: Decision

    pet_score: float | None = Score
    composite_score: float | None = Score

    ai_generated: bool | None = None
    ai_score: float | None = Score

    dimension_scores: DimensionScores | None = None

    hard_reject: bool = False
    rejection_code: RejectionCode | None = None
    flags: list[str] = Field(default_factory=list)
    species_detected: list[str] = Field(default_factory=list)

    processing_ms: int = Field(ge=0)
    model_version: str
