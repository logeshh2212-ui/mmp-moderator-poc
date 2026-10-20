import asyncio
from dataclasses import dataclass
from uuid import UUID

from redis.asyncio import Redis

from app.api.schemas import ModerationResponse
from app.core.config import settings


@dataclass(frozen=True)
class UploadKey:
    """Everything that can change the answer for an upload."""

    sha256: str
    user_id: UUID | str
    content_type: str
    context: str

    def suffix(self) -> str:
        return f"{self.sha256}:{self.user_id}:{self.content_type}:{self.context}"


def result_key(key: UploadKey) -> str:
    return f"mod:idem:{key.suffix()}"


def lock_key(key: UploadKey) -> str:
    return f"mod:lock:{key.suffix()}"


class IdempotencyStore:
    """Small wrapper around Redis so tests can swap in fakeredis."""

    def __init__(self, redis: Redis):
        self.redis = redis

    async def get_cached(self, key: UploadKey) -> ModerationResponse | None:
        raw = await self.redis.get(result_key(key))
        if raw is None:
            return None
        return ModerationResponse.model_validate_json(raw)

    async def store(self, key: UploadKey, response: ModerationResponse) -> None:
        await self.redis.set(
            result_key(key),
            response.model_dump_json(),
            ex=settings.redis.cache_ttl_seconds,
        )

    async def lock(self, key: UploadKey) -> bool:
        """SET NX EX: True if we got the lock, False if someone else holds it."""
        acquired = await self.redis.set(
            lock_key(key),
            "1",
            nx=True,
            ex=settings.redis.lock_ttl_seconds,
        )
        return bool(acquired)

    async def release(self, key: UploadKey) -> None:
        await self.redis.delete(lock_key(key))

    async def wait_for_result(self, key: UploadKey) -> ModerationResponse | None:
        """Another request holds the lock: re-check briefly for its stored answer."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + settings.redis.lock_wait_seconds

        while loop.time() < deadline:
            await asyncio.sleep(settings.redis.lock_poll_seconds)
            cached = await self.get_cached(key)
            if cached is not None:
                return cached
        return None

def make_redis() -> Redis:
    return Redis.from_url(
        settings.redis.url,
        socket_connect_timeout=settings.redis.connect_timeout_seconds,
        socket_timeout=settings.redis.socket_timeout_seconds,
    )


_redis: Redis | None = None


def get_idempotency_store() -> IdempotencyStore:
    """FastAPI dependency. Tests override this with a fakeredis-backed store."""
    global _redis
    if _redis is None:
        _redis = make_redis()
    return IdempotencyStore(_redis)

