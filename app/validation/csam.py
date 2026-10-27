import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Protocol

from app.core.config import settings


logger = logging.getLogger("mmp.safety")


class CsamCheckError(Exception):
    """The check could not run. Never a decision: the API answers 503 (try again)."""


@dataclass(frozen=True)
class CsamResult:
    checked: bool   # False means no real check happened (placeholder)
    match: bool
    provider: str


class CsamChecker(Protocol):
    def check(self, path: Path, sha256: str) -> CsamResult: ...


class PlaceholderCsamChecker:
    """Stands in for a real provider until the legal agreements exist.

    It never matches. It exists so the pipeline already has the right shape:
    the check runs after validation and before any model sees the file.
    """

    def __init__(self) -> None:
        logger.warning(
            "CSAM check is a PLACEHOLDER: uploads are NOT checked against any hash list. "
            "Do not run this in production."
        )

    def check(self, path: Path, sha256: str) -> CsamResult:
        return CsamResult(checked=False, match=False, provider="placeholder")


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HashListCsamChecker:
    """Exact matches against a list of known hashes (SHA-256 or MD5, hex, one per line).

    Only catches byte-identical copies; perceptual matching (PhotoDNA) needs a provider.
    The list comes from an official source through trust-and-safety and is kept outside
    the repo. An unreadable list stops the app at startup (fail-safe).
    """

    def __init__(self, hash_file: Path) -> None:
        try:
            lines = hash_file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise CsamCheckError(f"CSAM hash list unreadable: {exc}") from exc
        self.hashes = {line.strip().lower() for line in lines if line.strip() and not line.startswith("#")}
        self.check_md5 = any(len(h) == 32 for h in self.hashes)
        logger.info("CSAM hash list loaded: %d hashes", len(self.hashes))

    def check(self, path: Path, sha256: str) -> CsamResult:
        try:
            match = sha256.lower() in self.hashes or (self.check_md5 and file_md5(path) in self.hashes)
        except OSError as exc:
            raise CsamCheckError(f"CSAM check could not read the upload: {exc}") from exc
        return CsamResult(checked=True, match=match, provider="hash_list")


@cache
def get_csam_checker() -> CsamChecker:
    """Created once, selected by settings.safety.csam_provider."""
    safety = settings.safety
    if safety.csam_provider == "placeholder":
        return PlaceholderCsamChecker()
    if safety.csam_provider == "hash_list":
        return HashListCsamChecker(safety.hash_list_file)
    raise ValueError(f"Unknown CSAM provider: {safety.csam_provider}")


def quarantine(path: Path, record: dict) -> Path:
    """Move a matched upload into the quarantine folder with a JSON report record, so it is
    kept for the legal report instead of being deleted. Returns the new path."""
    folder = settings.safety.quarantine_dir
    folder.mkdir(parents=True, exist_ok=True)
    name = str(record["content_id"])
    target = folder / f"{name}{path.suffix.lower()}"
    shutil.move(str(path), target)
    record = {**record, "quarantined_at": datetime.now(timezone.utc).isoformat(), "file": target.name}
    (folder / f"{name}.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    logger.critical("CSAM match quarantined: content_id=%s provider=%s", name, record.get("provider"))
    return target
