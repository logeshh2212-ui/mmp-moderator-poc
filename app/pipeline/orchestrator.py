import logging
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from app.api.schemas import DimensionScores, ModerationResponse
from app.cache.blocklist import get_blocklist
from app.core.config import settings
from app.core.logging import log_result
from app.decision.engine import decide
from app.gates.detection import detect_animals, detector_frame_indices
from app.gates.gate1 import run_gate1
from app.gates.gate2 import FrameSignals, score_gate2
from app.gates.quality import frame_quality, jpeg_quality
from app.gates.relevance import score_relevance
from app.ml.registry import get_backbone, get_detector, get_species_model
from app.pipeline.context import ModerationContext
from app.prepare import prepare_media
from app.validation.csam import get_csam_checker
from app.validation.errors import ValidationFailure
from app.validation.formats import check_format
from app.validation.hashing import phash_image, phash_png_bytes
from app.validation.image_check import check_image
from app.validation.metadata import read_image_metadata, read_video_metadata
from app.validation.result import ValidationResult
from app.validation.video_check import check_video


gate1_logger = logging.getLogger("mmp.gate1")
gate2_logger = logging.getLogger("mmp.gate2")


@contextmanager
def timed(ctx: ModerationContext, stage: str):
    """Record how long a stage took (ms) in ctx.timings; logged on every request."""
    start = perf_counter()
    try:
        yield
    finally:
        ctx.timings[stage] = round((perf_counter() - start) * 1000)


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
        with timed(ctx, "validation"):
            ctx.validation = run_checks(ctx.path, ctx.content_type)
    except ValidationFailure as failure:
        ctx.reject("MOD_008", failure.flag)
        return finish(ctx)

    with timed(ctx, "safety"):
        # Step 8b: must run before any model or reviewer could see the file.
        ctx.csam = get_csam_checker().check(ctx.path, ctx.sha256)
        if not ctx.csam.match:
            # Step 10b: a re-upload of a known-bad image gets the earlier decision.
            phash = ctx.validation.phash
            ctx.blocklist_match = get_blocklist().find(phash) if phash else None
    if ctx.csam.match:
        # A real provider will also move the file to restricted quarantine
        # storage and start legal reporting before the temp file is deleted.
        ctx.reject("MOD_004", "quarantined", hard=True)
        return finish(ctx)
    if ctx.blocklist_match:
        entry = ctx.blocklist_match.entry
        ctx.reject(entry.rejection_code, "known_bad_image", hard=entry.hard_reject)
        return finish(ctx)

    # Graph Step 2 (Steps 13-14): decode and resize for the models.
    try:
        with timed(ctx, "prepare"):
            ctx.prepared = prepare_media(ctx.path, ctx.validation)
    except ValidationFailure as failure:
        ctx.reject("MOD_008", failure.flag)
        return finish(ctx)

    # Step 16: one SigLIP 2 pass per image / frame; Gate 1 (and later Gate 2) reuse it.
    backbone = get_backbone()
    if backbone is not None:
        with timed(ctx, "siglip"):
            ctx.embeddings = backbone.embed_images([frame.backbone for frame in ctx.prepared.frames])
            ctx.prompt_probs = backbone.probabilities(ctx.embeddings)

    ctx.flags.append("validation_passed")

    # Step 17-18: Gate 1 (spec section 2). Switched off only when no backbone is configured.
    if settings.ml.backbone:
        if backbone is None or (settings.ml.detector and get_detector() is None):
            ctx.flags.append("gate1_unavailable")  # a model failed to load: never pass or reject blindly
            return finish(ctx)
        run_gate1_branches(ctx, backbone)
        apply_gate1(ctx)
        # Gate 2 (spec 3-5) for everything Gate 1 did not rule out.
        if ctx.gate1.outcome != "fail":
            run_gate2(ctx, backbone)
            apply_gate2(ctx)

    # Later phases each add one call here (AI label, audit log).
    return finish(ctx)


def run_gate1_branches(ctx: ModerationContext, backbone) -> None:
    """SigLIP on every frame; detector + species model on the frames they can afford."""
    is_video = ctx.content_type == "video"
    sims = backbone.similarities(ctx.embeddings)
    detections = None

    detector = get_detector()
    if detector is not None:
        indices = detector_frame_indices(len(ctx.prepared.frames), is_video)
        pictures = [ctx.prepared.frames[i].detector for i in indices]
        species_model = get_species_model()
        with timed(ctx, "detector+species"):
            detections = detect_animals(pictures, detector, species_model)
        ctx.detections = dict(zip(indices, detections))  # Gate 2 measures D5 on these boxes
        sims = sims[indices]  # Gate 1 judges the frames both branches looked at

    with timed(ctx, "gate1"):
        ctx.gate1 = run_gate1(sims, backbone.labels, backbone.scale, is_video, detections)


def apply_gate1(ctx: ModerationContext) -> None:
    result = ctx.gate1
    ctx.pet_score = result.pet_score
    log_gate1(ctx)

    if result.outcome == "fail":
        if settings.gate1.enforce:
            ctx.reject("MOD_001", "no_pet_detected")
        else:
            ctx.flags.append("gate1_would_reject")  # shadow mode
        return

    ctx.species_detected = list(result.species)
    if result.outcome == "borderline":
        ctx.flags.append("gate1_borderline")
        return

    ctx.flags.append("gate1_passed")
    if result.wild:
        ctx.flags.append("wild_context")  # decided: wild / marine animals go to a human


def source_info(ctx: ModerationContext) -> tuple[int, float | None]:
    """Short side of the original upload, and the JPEG save quality (images only)."""
    image, video = ctx.validation.image, ctx.validation.video
    if image is not None:
        return min(image.width, image.height), jpeg_quality(ctx.path)
    return min(video.width, video.height), None


def run_gate2(ctx: ModerationContext, backbone) -> None:
    """D5 on every frame (on Gate 1's pet box where the detector looked) and D6 from the
    SigLIP embeddings. D1-D4 detectors arrive in steps 2d / 2e; until then no signals."""
    with timed(ctx, "gate2"):
        short_side, jpeg_q = source_info(ctx)
        for i, frame in enumerate(ctx.prepared.frames):
            found = ctx.detections.get(i)
            box = found.box if found else None
            ctx.quality.append(frame_quality(frame.detector, box, short_side, jpeg_q))

        sims = backbone.similarities(ctx.embeddings)
        ctx.relevance = score_relevance(sims, backbone.labels, backbone.scale) or []
        relevance = [r.score for r in ctx.relevance] or [None] * len(ctx.quality)

        frames = [FrameSignals(quality=q.score, relevance=r) for q, r in zip(ctx.quality, relevance)]
        ctx.gate2 = score_gate2(frames)
        ctx.verdict = decide(ctx.gate2)


def apply_gate2(ctx: ModerationContext) -> None:
    verdict = ctx.verdict
    ctx.composite_score = verdict.composite_score
    dimensions = ctx.gate2.dimensions
    if all(value is not None for value in dimensions.values()):
        ctx.dimension_scores = DimensionScores(**{d: round(v, 1) for d, v in dimensions.items()})
    ctx.flags += verdict.flags
    log_gate2(ctx)

    if verdict.decision == "REJECTED":
        if settings.gate2.enforce:
            ctx.reject(verdict.rejection_code, "gate2_rejected", hard=verdict.hard_reject)
        else:
            ctx.flags.append("gate2_would_reject")  # shadow mode
        return
    if verdict.decision == "HUMAN_REVIEW":
        ctx.flags.append("gate2_review")
        return
    if ctx.gate1.outcome != "pass" or ctx.gate1.wild:
        return  # Gate 1 already sent it to a human
    if settings.gate1.enforce and settings.gate2.enforce and settings.gate1.approve_on_pass:
        ctx.flags.append("d1_d4_unchecked")  # POC: no safety detectors yet (steps 2d / 2e)
        ctx.decision = "APPROVED"


def log_gate2(ctx: ModerationContext) -> None:
    """One line per upload with the six dimensions and what D5 / D6 saw, for checking by hand."""
    d = ctx.gate2.dimensions

    def show(value):
        return "-" if value is None else f"{value:.0f}"

    issue = Counter(q.issue for q in ctx.quality).most_common(1)[0][0]
    kind = Counter(r.kind for r in ctx.relevance).most_common(1)[0][0] if ctx.relevance else "-"
    sure = max((r.confidence for r in ctx.relevance if r.kind == kind), default=0.0)
    gate2_logger.info(
        "content_id=%s verdict=%s composite=%s D1=%s D2=%s D3=%s D4=%s D5=%s (%s) D6=%s (%s %.2f) "
        "signals=%s frames=%d",
        ctx.content_id, ctx.verdict.decision, show(ctx.verdict.composite_score),
        show(d["D1_welfare"]), show(d["D2_nsfw"]), show(d["D3_violence"]), show(d["D4_prohibited"]),
        show(d["D5_quality"]), issue, show(d["D6_relevance"]), kind, sure,
        ",".join(ctx.gate2.fired) or "-", len(ctx.quality),
    )


def log_gate1(ctx: ModerationContext) -> None:
    """One line per upload with what each Gate 1 branch saw, for checking results by hand."""
    result = ctx.gate1
    top = max(result.frames, key=lambda f: f.pet_score)
    kind = "drawing" if top.illustrated else "insect" if top.insect else "photo"
    detector = "off" if top.det_max is None else f"{top.det_label or '-'} ({top.det_max:.2f})"
    gate1_logger.info(
        "content_id=%s pet_score=%d outcome=%s bar=%.0f kind=%s siglip=%.0f detector=%s "
        "lookalike=%s species=%s best_animal=%s (%.3f) best_no_pet=%s (%.3f) wild=%s frames=%d "
        "passed=%.0f%%",
        ctx.content_id, result.pet_score, result.outcome, top.threshold, kind, top.siglip_score,
        detector, top.lookalike or "-", ",".join(result.species) or "-", top.best_animal,
        top.animal_similarity,
        top.best_no_pet, top.no_pet_similarity, result.wild, len(result.frames),
        result.pass_fraction * 100,
    )


def finish(ctx: ModerationContext) -> ModerationResponse:
    remember_if_bad(ctx)
    response = ctx.to_response()
    log_result(response, timings=ctx.timings)
    return response


def remember_if_bad(ctx: ModerationContext) -> None:
    """Add this image to the blocklist if its rejection code is one we block automatically."""
    phash = ctx.validation.phash if ctx.validation else None
    if (
        ctx.decision == "REJECTED"
        and phash
        and ctx.blocklist_match is None  # already on the list
        and ctx.rejection_code in settings.blocklist.auto_block_codes
    ):
        get_blocklist().add(phash, ctx.rejection_code, hard_reject=ctx.hard_reject, source="auto")
