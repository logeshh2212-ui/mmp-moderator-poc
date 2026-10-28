from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from uuid import UUID

import numpy as np

from app.api.schemas import Decision, DetectedObject, DimensionScores, ModerationResponse
from app.core.versioning import current_model_version
from app.validation.result import ValidationResult
from app.validation.csam import CsamResult
from app.cache.blocklist import BlocklistMatch
from app.prepare.media import PreparedMedia
from app.gates.gate1 import FrameDetections, Gate1Result
from app.gates.gate2 import Gate2Result
from app.gates.quality import QualityResult
from app.gates.relevance import RelevanceResult
from app.gates.audio import AudioResult
from app.gates.scene import SceneResult
from app.decision.engine import Verdict



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

    # Graph Step 2: pictures at model sizes, for Gate 1 and later
    prepared: PreparedMedia | None = None

    # Step 16: SigLIP 2, one embedding per image / frame (reused by Gate 1 and Gate 2)
    embeddings: np.ndarray | None = None
    prompt_probs: dict[str, list[float]] | None = None

    # Step 17: Gate 1
    gate1: Gate1Result | None = None
    detections: dict[int, FrameDetections] = field(default_factory=dict)  # frame index -> boxes

    # Gate 2: per-frame D1-D3 scenes, NSFW probability, D5 / D6, then the dimension
    # scores and the engine's verdict
    scene: list[SceneResult] = field(default_factory=list)
    nsfw: dict[int, float] = field(default_factory=dict)  # frame index -> nsfw probability
    ocr_text: dict[int, str] = field(default_factory=dict)  # frame index -> text read on it
    audio: AudioResult | None = None                        # video soundtrack (spec 6.2)
    quality: list[QualityResult] = field(default_factory=list)
    relevance: list[RelevanceResult] = field(default_factory=list)
    gate2: Gate2Result | None = None
    verdict: Verdict | None = None

    # Milliseconds per pipeline stage, logged on every request (speed work comes later)
    timings: dict[str, int] = field(default_factory=dict)

    # Filled in by later phases (Gate 1, Gate 2, scoring, AI label)
    pet_score: float | None = None
    composite_score: float | None = None
    dimension_scores: DimensionScores | None = None
    species_detected: list[str] = field(default_factory=list)
    objects_detected: list[DetectedObject] = field(default_factory=list)
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
            objects_detected=list(self.objects_detected),
            processing_ms=int((perf_counter() - self.started_at) * 1000),
            model_version=current_model_version(),
        )
