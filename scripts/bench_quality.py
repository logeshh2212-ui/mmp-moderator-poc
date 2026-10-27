"""Measure D5 on a folder made by make_quality_set: raw measures and score per kind.

    uv run python -m scripts.bench_quality D:/samples/quality
    uv run python -m scripts.bench_quality D:/samples/quality --details   (one line per file)

Runs the real validation and pre-processing, then app.gates.quality on the whole picture
(no Gate 1 box, so no models are loaded). Expected bands come from spec 4.5.
"""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from app.core.config import settings
from app.gates.quality import frame_quality, jpeg_quality
from app.pipeline.orchestrator import run_checks
from app.prepare import prepare_media

EXPECTED = {
    "good": "sharp", "slight_blur": "slight_blur", "heavy_blur": "heavy_blur",
    "dark": "dark", "overexposed": "dark", "compressed": "compressed",
    "lowres": "screenshot", "screenshot": "screenshot", "blank": None,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    bands = settings.gate2.quality.bands

    for kind_dir in sorted(p for p in args.folder.iterdir() if p.is_dir()):
        kind = kind_dir.name
        band = getattr(bands, EXPECTED.get(kind) or "", None) or ((0, 0) if kind == "blank" else None)
        rows = []
        for path in sorted(kind_dir.iterdir()):
            start = perf_counter()
            validation = run_checks(path, "image")
            frame = prepare_media(path, validation).frames[0]
            image = validation.image
            result = frame_quality(frame.detector, None, min(image.width, image.height), jpeg_quality(path))
            rows.append((path.name, result, (perf_counter() - start) * 1000))

        scores = np.array([r.score for _, r, _ in rows])
        inside = np.mean([band[0] <= s <= band[1] for s in scores]) if band else float("nan")
        sharp = np.array([r.measures.sharpness for _, r, _ in rows])
        issues = {}
        for _, r, _ in rows:
            issues[r.issue] = issues.get(r.issue, 0) + 1
        print(f"{kind:<12} n={len(rows):<3} score p10/p50/p90 {np.percentile(scores, 10):5.1f} "
              f"{np.median(scores):5.1f} {np.percentile(scores, 90):5.1f}  in spec band {band}: "
              f"{inside:4.0%}  sharpness p10/p50/p90 {np.percentile(sharp, 10):7.1f} "
              f"{np.median(sharp):7.1f} {np.percentile(sharp, 90):7.1f}  issues {issues}")
        if args.details:
            for name, r, ms in rows:
                m = r.measures
                print(f"    {r.score:5.1f} {r.issue:<11} sharp {m.sharpness:7.1f} contrast {m.contrast:5.1f} "
                      f"bright {m.brightness:5.1f} burnt {m.bright_fraction:4.2f} bars {m.bar_fraction:4.2f} "
                      f"side {m.short_side} jpegq {m.jpeg_quality or 0:4.0f}  {ms:4.0f} ms  {name[:40]}")


if __name__ == "__main__":
    main()
