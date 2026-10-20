"""Manage the pHash blocklist by hand.

Usage (from the project folder, Redis running):
    uv run python -m scripts.blocklist add    D:/samples/bad.jpg image MOD_003 [--hard]
    uv run python -m scripts.blocklist check  D:/samples/copy.jpg image
    uv run python -m scripts.blocklist remove D:/samples/bad.jpg image
    uv run python -m scripts.blocklist count
"""
import argparse
import getpass
import sys
from pathlib import Path
from typing import get_args

from app.api.schemas import RejectionCode
from app.cache.blocklist import get_blocklist
from app.pipeline.orchestrator import run_checks
from app.validation.errors import ValidationFailure


def phash_of(path: str, content_type: str) -> str:
    try:
        phash = run_checks(Path(path), content_type).phash
    except ValidationFailure as failure:
        sys.exit(f"File fails validation ({failure.flag}); nothing to hash.")
    if not phash:
        sys.exit("Could not compute a perceptual hash for this file.")
    return phash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add")
    add.add_argument("path")
    add.add_argument("content_type", choices=["image", "video"])
    add.add_argument("code", choices=get_args(RejectionCode))
    add.add_argument("--hard", action="store_true", help="mark as hard reject")

    for name in ("check", "remove"):
        cmd = sub.add_parser(name)
        cmd.add_argument("path")
        cmd.add_argument("content_type", choices=["image", "video"])

    sub.add_parser("count")
    args = parser.parse_args()
    blocklist = get_blocklist()

    if args.command == "count":
        print(f"{blocklist.count()} blocked hashes")
        return

    phash = phash_of(args.path, args.content_type)

    if args.command == "add":
        if args.code == "MOD_008":
            sys.exit("MOD_008 means a broken file, not bad content; it does not belong on the blocklist.")
        blocklist.add(phash, args.code, hard_reject=args.hard, source=f"manual:{getpass.getuser()}")
        print(f"added {phash} -> {args.code}")
    elif args.command == "check":
        match = blocklist.find(phash)
        print(f"{phash}: " + (f"BLOCKED as {match.entry.rejection_code} (distance {match.distance})" if match else "not blocked"))
    elif args.command == "remove":
        print(f"{phash}: " + ("removed" if blocklist.remove(phash) else "was not on the list"))


if __name__ == "__main__":
    main()
