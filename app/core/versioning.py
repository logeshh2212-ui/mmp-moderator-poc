import hashlib
import json

from app.core.config import settings


def decision_settings() -> dict:
    """Every setting that can change a moderation decision.

    Runtime-only settings (Redis, timeouts, paths, logging, threads) are left out, so they
    can change without invalidating cached answers.
    """
    ml = settings.ml
    backbone = ml.backbone
    return {
        "limits": settings.limits.model_dump(),
        "allowed_formats": settings.allowed_formats.model_dump(),
        "blocklist": settings.blocklist.model_dump(),
        "prepare": settings.prepare.model_dump(exclude={"ffmpeg_hwaccel"}),
        "ml": {
            "backbone": backbone,
            "dtype": ml.dtype,
            "model": ml.models[backbone].model_dump() if backbone else None,
            "detector": ml.models[ml.detector].model_dump() if ml.detector else None,
            "species_model": ml.models[ml.species_model].model_dump() if ml.species_model else None,
            "prompts": ml.prompts,
        },
        "gate1": settings.gate1.model_dump(),
        "gate2": settings.gate2.model_dump(),
    }


def settings_fingerprint() -> str:
    """Short hash of the decision settings: changes whenever any of them changes."""
    blob = json.dumps(decision_settings(), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def current_model_version() -> str:
    """Logged on every decision (spec 8.3) and part of the cache key.

    Example: validation-0.1+gate1-fused-0.4+gate2-0.2+siglip2_base@f775b65+cfg3a9c41d2
    """
    backbone = settings.ml.backbone
    model = (
        f"{backbone}@{settings.ml.models[backbone].revision[:7]}" if backbone else "no-model"
    )
    return "+".join([
        settings.model_version,
        settings.gate1.version,
        settings.gate2.version,
        model,
        f"cfg{settings_fingerprint()}",
    ])
