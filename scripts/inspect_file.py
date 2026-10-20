"""Run validation on one file and print everything it found.

Usage (from the project folder):
    uv run python -m scripts.inspect_file D:/samples/dog.jpg image
    uv run python -m scripts.inspect_file D:/samples/dog.jpg image --json
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path
from pprint import pprint

from app.pipeline.orchestrator import run_checks
from app.validation.errors import ValidationFailure


def main() -> None:
    path, content_type = Path(sys.argv[1]), sys.argv[2]
    as_json = "--json" in sys.argv[3:]

    try:
        data = asdict(run_checks(path, content_type))
    except ValidationFailure as failure:
        data = {"rejected": failure.flag}

    if data.get("video"):
        frame = data["video"]["first_frame_png"]
        data["video"]["first_frame_png"] = f"<{len(frame)} bytes of PNG>"

    if as_json:
        print(json.dumps(data, default=str))
    elif "rejected" in data:
        print(f"REJECTED: {data['rejected']}")
    else:
        pprint(data, sort_dicts=False, width=100)


if __name__ == "__main__":
    main()
