"""Run validation on one file and print everything it found.

Usage (from the project folder):
    uv run python -m scripts.inspect_file D:/samples/dog.jpg image
"""
import sys
from dataclasses import asdict
from pathlib import Path
from pprint import pprint

from app.api.routes import run_checks
from app.validation.errors import ValidationFailure


def main() -> None:
    path, content_type = Path(sys.argv[1]), sys.argv[2]

    try:
        result = run_checks(path, content_type)
    except ValidationFailure as failure:
        print(f"REJECTED: {failure.flag}")
        return

    data = asdict(result)
    if data["video"]:
        frame = data["video"]["first_frame_png"]
        data["video"]["first_frame_png"] = f"<{len(frame)} bytes of PNG>"
    pprint(data, sort_dicts=False, width=100)


if __name__ == "__main__":
    main()
