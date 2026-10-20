"""Benchmark SigLIP 2 on this machine against the spec targets (image < 3 s, 60 s video < 15 s, p95).

    uv run python -m scripts.bench_models --images D:/samples/real --video D:/samples/bench_60s_1080p.mp4
    uv run python -m scripts.bench_models --models siglip2_base --threads 4 --runs 2
"""
import argparse
import gc
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from app.core.config import settings
from app.ml.siglip import SiglipEmbedder
from app.pipeline.orchestrator import run_checks
from app.prepare import prepare_media
from app.validation.errors import ValidationFailure

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def end_to_end(model, path, content_type):
    """What one upload costs: validation + pre-processing, then SigLIP (timed separately)."""
    start = time.perf_counter()
    prepared = prepare_media(path, run_checks(path, content_type))
    middle = time.perf_counter()
    embeds = model.embed_images([f.backbone for f in prepared.frames])
    return model.probabilities(embeds), len(prepared.frames), middle - start, time.perf_counter() - middle



def pct(values, q) -> float:
    return float(np.percentile(values, q))


def mark(seconds, limit) -> str:
    return "OK  " if seconds < limit else "MISS"


def valid_images(folder: Path | None) -> list[Path]:
    if folder is None:
        return []
    good = []
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT):
        try:
            run_checks(path, "image")
            good.append(path)
        except ValidationFailure:
            pass
    return good


def bench(name, images, video, threads_list, runs):
    gc.collect()
    before = rss_gb()
    model = SiglipEmbedder(name)
    _, load_s = timed(model.load)
    model.warmup()
    print(f"\n=== {name}  ({model.spec.repo} @ {model.spec.revision[:7]}) ===")
    print(f"load {load_s:.1f} s · memory +{rss_gb() - before:.2f} GB (text half dropped) · {model.dim}-d")

    for threads in threads_list:
        torch.set_num_threads(threads)
        print(f"\n-- {threads} thread(s)")

        if images:
            pictures = [f.backbone for f in prepare_media(images[0], run_checks(images[0], "image")).frames] * 33
            for size in (1, 8, 33):
                print(f"   throughput at batch 33: {33 / s:.1f} pictures/s (spec: 50 images/s)")
                settings.ml.batch_size = size
                _, s = timed(model.embed_images, pictures[:size])
                print(f"   SigLIP only, batch {size:>2}: {s * 1000 / size:7.0f} ms per picture")
            settings.ml.batch_size = 16

            times = []
            for _ in range(runs):
                for path in images:
                    _, s = timed(end_to_end, model, path, "image")
                    times.append(s)
            p95 = pct(times, 95)
            print(f"   image end-to-end ({len(times)} runs): p50 {pct(times, 50):.2f} s · "
                  f"p95 {p95:.2f} s  [{mark(p95, settings.targets.image_p95_seconds)} < "
                  f"{settings.targets.image_p95_seconds:g} s]")

        if video:
            times, preps, models_s, frames = [], [], [], 0
            for _ in range(max(1, runs // 2)):
                (_, frames, prep_s, model_s), s = timed(end_to_end, model, video, "video")
                times.append(s)
                preps.append(prep_s)
                models_s.append(model_s)
            p95 = pct(times, 95)
            print(f"   video end-to-end ({frames} frames, {len(times)} runs): p50 {pct(times, 50):.1f} s · "
                  f"p95 {p95:.1f} s  [{mark(p95, settings.targets.video_p95_seconds)} < "
                  f"{settings.targets.video_p95_seconds:g} s]")
            print(f"      of which pre-processing {np.mean(preps):.1f} s, SigLIP {np.mean(models_s):.1f} s")


    if images:
        torch.set_num_threads(settings.ml.num_threads)
        print("\n-- sanity: best-matching sentence per image")
        for path in images:
            probs, *_ = end_to_end(model, path, "image")
            best = max(probs, key=lambda label: probs[label][0])
            others = ", ".join(f"{label} {values[0]:.2f}" for label, values in probs.items())
            print(f"   {path.name:<28} -> {best:<10} ({others})")

    del model
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(settings.ml.models))
    parser.add_argument("--images", type=Path, help="folder of real photos")
    parser.add_argument("--video", type=Path, help="a 60 s video")
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    images = valid_images(args.images)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"torch {torch.__version__} · device {settings.ml.device} ({settings.ml.dtype}) · GPU {gpu} · "
          f"{len(images)} valid images · video: {args.video or '-'}")

    for name in args.models:
        bench(name, images, args.video, args.threads, args.runs)


if __name__ == "__main__":
    main()
