from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache

from redis import Redis

from app.core.config import settings
from app.validation.hashing import hamming


BANDS = 8  # a 16-hex-character pHash split into 8 bands of 2 characters (8 bits each)


def entry_key(phash: str) -> str:
    return f"mod:phash:entry:{phash}"


def band_key(index: int, band: str) -> str:
    return f"mod:phash:band:{index}:{band}"


def bands(phash: str) -> list[str]:
    return [phash[i * 2 : i * 2 + 2] for i in range(BANDS)]


@dataclass(frozen=True)
class BlocklistEntry:
    phash: str
    rejection_code: str
    hard_reject: bool
    source: str      # "auto" or who added it by hand
    added_at: str


@dataclass(frozen=True)
class BlocklistMatch:
    entry: BlocklistEntry
    distance: int


class PhashBlocklist:
    """Known-bad images by perceptual hash, stored in Redis. Synchronous on purpose:
    it runs inside the pipeline's worker thread."""

    def __init__(self, redis: Redis):
        self.redis = redis

    def add(self, phash: str, rejection_code: str, *, hard_reject: bool = False, source: str = "auto") -> None:
        pipe = self.redis.pipeline()
        pipe.hset(entry_key(phash), mapping={
            "rejection_code": rejection_code,
            "hard_reject": "1" if hard_reject else "0",
            "source": source,
            "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        for index, band in enumerate(bands(phash)):
            pipe.sadd(band_key(index, band), phash)
        pipe.execute()

    def remove(self, phash: str) -> bool:
        pipe = self.redis.pipeline()
        pipe.delete(entry_key(phash))
        for index, band in enumerate(bands(phash)):
            pipe.srem(band_key(index, band), phash)
        return bool(pipe.execute()[0])

    def get(self, phash: str) -> BlocklistEntry | None:
        data = self.redis.hgetall(entry_key(phash))
        if not data:
            return None
        return BlocklistEntry(
            phash=phash,
            rejection_code=data["rejection_code"],
            hard_reject=data["hard_reject"] == "1",
            source=data["source"],
            added_at=data["added_at"],
        )

    def find(self, phash: str) -> BlocklistMatch | None:
        """Closest blocked hash within max_distance, or None."""
        pipe = self.redis.pipeline()
        for index, band in enumerate(bands(phash)):
            pipe.smembers(band_key(index, band))
        candidates = set().union(*pipe.execute())

        best_hash, best_distance = None, settings.blocklist.max_distance + 1
        for candidate in candidates:
            distance = hamming(phash, candidate)
            if distance < best_distance:
                best_hash, best_distance = candidate, distance

        if best_hash is None:
            return None
        entry = self.get(best_hash)
        return BlocklistMatch(entry=entry, distance=best_distance) if entry else None

    def count(self) -> int:
        return sum(1 for _ in self.redis.scan_iter(match="mod:phash:entry:*"))


@cache
def get_blocklist() -> PhashBlocklist:
    """Created once. Tests replace this with an in-memory fake."""
    return PhashBlocklist(
        Redis.from_url(
            settings.redis.url,
            socket_connect_timeout=settings.redis.connect_timeout_seconds,
            socket_timeout=settings.redis.socket_timeout_seconds,
            decode_responses=True,
        )
    )
