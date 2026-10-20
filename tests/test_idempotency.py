import asyncio
from uuid import uuid4

import fakeredis

from app.api.schemas import ModerationResponse
from app.cache.idempotency import IdempotencyStore, lock_key, result_key
from app.core.config import settings


SHA = "a" * 64
USER = uuid4()


def make_response() -> ModerationResponse:
    return ModerationResponse(
        content_id=uuid4(),
        decision="HUMAN_REVIEW",
        processing_ms=5,
        model_version="test",
    )


def new_store() -> IdempotencyStore:
    return IdempotencyStore(fakeredis.FakeAsyncRedis())


def test_store_then_get_returns_same_response():
    async def scenario():
        store = new_store()
        response = make_response()

        assert await store.get_cached(SHA, USER) is None
        await store.store(SHA, USER, response)
        assert await store.get_cached(SHA, USER) == response

    asyncio.run(scenario())


def test_other_user_does_not_hit_cache():
    async def scenario():
        store = new_store()
        await store.store(SHA, USER, make_response())

        assert await store.get_cached(SHA, uuid4()) is None

    asyncio.run(scenario())


def test_stored_response_expires_after_cache_ttl():
    async def scenario():
        store = new_store()
        await store.store(SHA, USER, make_response())

        ttl = await store.redis.ttl(result_key(SHA, USER))
        assert 0 < ttl <= settings.redis.cache_ttl_seconds

    asyncio.run(scenario())


def test_lock_is_exclusive_until_released():
    async def scenario():
        store = new_store()

        assert await store.lock(SHA, USER) is True
        assert await store.lock(SHA, USER) is False

        ttl = await store.redis.ttl(lock_key(SHA, USER))
        assert 0 < ttl <= settings.redis.lock_ttl_seconds

        await store.release(SHA, USER)
        assert await store.lock(SHA, USER) is True

    asyncio.run(scenario())


def test_wait_for_result_returns_answer_stored_by_other_request(monkeypatch):
    monkeypatch.setattr(settings.redis, "lock_poll_seconds", 0.01)
    monkeypatch.setattr(settings.redis, "lock_wait_seconds", 1)

    async def scenario():
        store = new_store()
        response = make_response()

        async def other_request_finishes():
            await asyncio.sleep(0.05)
            await store.store(SHA, USER, response)

        asyncio.create_task(other_request_finishes())
        assert await store.wait_for_result(SHA, USER) == response

    asyncio.run(scenario())


def test_wait_for_result_gives_up_after_wait_time(monkeypatch):
    monkeypatch.setattr(settings.redis, "lock_poll_seconds", 0.01)
    monkeypatch.setattr(settings.redis, "lock_wait_seconds", 0.05)

    async def scenario():
        assert await new_store().wait_for_result(SHA, USER) is None

    asyncio.run(scenario())
