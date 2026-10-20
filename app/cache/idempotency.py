import asyncio
from uuid import UUID
from redis.asyncio import Redis
from app.api.schemas import ModerationResponse
from app.core.config import settings


def result_key(sha256: str, user_id: UUID | str) -> str:
    return f"mod:idem:{sha256}:{user_id}"


def lock_key(sha256: str, user_id: UUID | str) -> str:
    return f"mod:lock:{sha256}:{user_id}"


class IdempotencyStore:
    """Small wrapper around Redis so tests can swap in fakeredis."""

    def __init__(self, redis: Redis):
        self.redis = redis

    async def get_cached(self, sha256: str, user_id: UUID) -> ModerationResponse | None:
        raw = await self.redis.get(result_key(sha256, user_id))
        if raw is None:
            return None
        return ModerationResponse.model_validate_json(raw)

    async def store(self, sha256: str, user_id: UUID, response: ModerationResponse) -> None:
        await self.redis.set(
            result_key(sha256, user_id),
            response.model_dump_json(),
            ex=settings.redis.cache_ttl_seconds,
        )

    async def lock(self, sha256: str, user_id: UUID) -> bool:
        """SET NX EX: True if we got the lock, False if someone else holds it."""
        acquired = await self.redis.set(
            lock_key(sha256, user_id),
            "1",
            nx=True,
            ex=settings.redis.lock_ttl_seconds,
        )
        return bool(acquired)

    async def release(self, sha256: str, user_id: UUID) -> None:
        await self.redis.delete(lock_key(sha256, user_id))

    async def wait_for_result(self, sha256: str, user_id: UUID) -> ModerationResponse | None:
        """Another request holds the lock: re-check briefly for its stored answer."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + settings.redis.lock_wait_seconds

        while loop.time() < deadline:
            await asyncio.sleep(settings.redis.lock_poll_seconds)
            cached = await self.get_cached(sha256, user_id)
            if cached is not None:
                return cached
        return None


_redis = Redis.from_url(
    settings.redis.url,
    socket_connect_timeout=2,
    socket_timeout=2,
)



def get_idempotency_store() -> IdempotencyStore:
    """FastAPI dependency. Tests override this with a fakeredis-backed store."""
    global _redis
    if _redis is None:
        _redis = Redis.from_url(settings.redis.url)
    return IdempotencyStore(_redis)
