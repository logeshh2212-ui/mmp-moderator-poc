import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Protocol

from app.core.config import settings


logger = logging.getLogger("mmp.safety")


@dataclass(frozen=True)
class CsamResult:
    checked: bool   # False means no real check happened (placeholder)
    match: bool
    provider: str


class CsamChecker(Protocol):
    def check(self, path: Path, sha256: str) -> CsamResult: ...


class PlaceholderCsamChecker:
    """Stands in for PhotoDNA / Thorn Safer until the legal agreements exist.

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


@cache
def get_csam_checker() -> CsamChecker:
    """Created once. A real provider is added here, selected by settings.safety.csam_provider."""
    if settings.safety.csam_provider == "placeholder":
        return PlaceholderCsamChecker()
    raise ValueError(f"Unknown CSAM provider: {settings.safety.csam_provider}")
