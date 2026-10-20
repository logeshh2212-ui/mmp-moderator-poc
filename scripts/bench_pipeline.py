"""Run the REAL moderation pipeline (all Gate 1 models) on a folder: decisions, pet_score,
species and time per stage, with p50 / p95 against the spec targets.

Folder layout: sub-folders named after the expected result.
    pet/      -> should be APPROVED
    no_pet/   -> should be REJECTED
    review/   -> should be HUMAN_REVIEW
    (any other sub-folder name is run but not judged)

    uv run python -m scripts.bench_pipeline D:/samples/gate1
    ML_DEVICE=cuda ML_DTYPE=float16 python -m scripts.bench_pipeline ../inputs/gate1 --runs 2   (Colab)

No Redis is needed: the duplicate check and the blocklist are skipped on purpose.
"""
import argparse
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch

from app.core.config import settings
from app.core.versioning import current_model_version
from app.ml.registry import load_models, registry
from app.pipeline import orchestrator
from app.pipeline.context import ModerationContext

EXPECTED = {"pet": "APPROVED", "no_pet": "REJECTED", "review": "HUMAN_REVIEW"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
VIDEO_EXT = {".mp4", ".mov", ".avi"}


class NoBlocklist:
    """Benchmark only: no Redis. Nothing is looked up or remembered."""

    def find(self, phash):
        return None

    def add(self, *args, **kwargs):
        pass


def moderate(path: Path, content_type: str):
    work = Path(tempfile.mkdtemp())
    copy = work / path.name
    shutil.copy(path, copy)
    ctx = ModerationContext(
        content_id=uuid4(), user_id=uuid4(), content_type=content_type, context="post",
        started_at=perf_counter(), path=copy, sha256=uuid4().hex + uuid4().hex,
    )
    try:
        response = orchestrator.run_pipeline(ctx)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return response, ctx.timings


def pct(values, q) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--runs", type=int, default=1, help="runs per file (p95 needs several)")
    args = parser.parse_args()

    orchestrator.get_blocklist = lambda: NoBlocklist()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"device {settings.ml.device} ({settings.ml.dtype}) · GPU {gpu} · torch {torch.__version__}")
    start = perf_counter()
    load_models()
    print(f"models loaded in {perf_counter() - start:.1f} s: {registry.status}")
    print(f"model_version {current_model_version()}\n")

    totals = {"image": [], "video": []}
    stages: dict[str, list[int]] = {}
    judged = correct = 0
    for folder in sorted(p for p in args.folder.iterdir() if p.is_dir()):
        expected = EXPECTED.get(folder.name)
        for path in sorted(folder.iterdir()):
            ext = path.suffix.lower()
            if ext not in IMAGE_EXT | VIDEO_EXT:
                continue
            content_type = "video" if ext in VIDEO_EXT else "image"
            for _ in range(args.runs):
                begin = perf_counter()
                response, timings = moderate(path, content_type)
                totals[content_type].append(perf_counter() - begin)
                for stage, ms in timings.items():
                    stages.setdefault(f"{content_type}:{stage}", []).append(ms)

            mark = "  "
            if expected:
                ok = response.decision == expected
                judged += 1
                correct += ok
                mark = "OK" if ok else "XX"
            species = ",".join(response.species_detected) or "-"
            stage_text = " ".join(f"{s}:{ms}" for s, ms in timings.items())
            print(f"{mark} {folder.name:<7} {response.decision:<12} {response.pet_score!s:>5} "
                  f"{species:<20} {path.name[:36]:<36} | {stage_text}")

    print("\n--- decisions ---")
    if judged:
        print(f"correct {correct}/{judged} ({correct / judged:.0%})")

    print("\n--- total time per upload (spec: image < 3 s, 60 s video < 15 s, p95) ---")
    limits = {"image": settings.targets.image_p95_seconds, "video": settings.targets.video_p95_seconds}
    for content_type, values in totals.items():
        if values:
            p95 = pct(values, 95)
            verdict = "OK" if p95 < limits[content_type] else "MISS"
            print(f"{content_type:<6} n={len(values):<3} p50 {pct(values, 50):6.2f} s · p95 {p95:6.2f} s  [{verdict} < {limits[content_type]:g} s]")

    print("\n--- average ms per stage ---")
    for name, values in sorted(stages.items()):
        print(f"{name:<28} {np.mean(values):8.0f}")


if __name__ == "__main__":
    main()
