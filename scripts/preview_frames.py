"""Save the prepared pictures (what the models will see) so you can check them by eye.

    uv run python -m scripts.preview_frames D:/samples/dog1.mp4 video
    uv run python -m scripts.preview_frames D:/samples/dog.jpg image D:/samples/out
"""
import sys
import time
from pathlib import Path

from app.pipeline.orchestrator import run_checks
from app.prepare import prepare_media
from app.validation.errors import ValidationFailure


def main() -> None:
    path, content_type = Path(sys.argv[1]), sys.argv[2]
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else path.with_name(f"{path.stem}_frames")

    try:
        validation = run_checks(path, content_type)
    except ValidationFailure as failure:
        sys.exit(f"REJECTED: {failure.flag}")

    start = time.perf_counter()
    prepared = prepare_media(path, validation)
    ms = round((time.perf_counter() - start) * 1000)

    out.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(prepared.frames):
        label = f"{i:02d}" if frame.timestamp is None else f"{i:02d}_t{frame.timestamp:06.2f}s"
        frame.detector.save(out / f"{label}_detector.png")
        frame.backbone.save(out / f"{label}_backbone.png")

    first = prepared.frames[0]
    print(f"{len(prepared.frames)} frame(s) prepared in {ms} ms -> {out}")
    print(f"detector {first.detector.size}, backbone {first.backbone.size}")
    if first.timestamp is not None:
        print("times:", ", ".join(f"{f.timestamp:.2f}" for f in prepared.frames))


if __name__ == "__main__":
    main()
