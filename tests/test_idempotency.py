import asyncio
from uuid import uuid4

import fakeredis

from app.api.schemas import ModerationResponse
from app.cache.idempotency import IdempotencyStore, UploadKey, lock_key, result_key
from app.core.config import settings
from app.cache.idempotency import make_redis

SHA = "a" * 64
USER = uuid4()
KEY = UploadKey(SHA, USER, "image", "post", "mv-test")


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

        assert await store.get_cached(KEY) is None
        await store.store(KEY, response)
        assert await store.get_cached(KEY) == response

    asyncio.run(scenario())


def test_other_user_does_not_hit_cache():
    async def scenario():
        store = new_store()
        await store.store(KEY, make_response())

        assert await store.get_cached(UploadKey(SHA, uuid4(), "image", "post", "mv-test")) is None

    asyncio.run(scenario())


def test_stored_response_expires_after_cache_ttl():
    async def scenario():
        store = new_store()
        await store.store(KEY, make_response())

        ttl = await store.redis.ttl(result_key(KEY))
        assert 0 < ttl <= settings.redis.cache_ttl_seconds

    asyncio.run(scenario())


def test_lock_is_exclusive_until_released():
    async def scenario():
        store = new_store()

        assert await store.lock(KEY) is True
        assert await store.lock(KEY) is False

        ttl = await store.redis.ttl(lock_key(KEY))
        assert 0 < ttl <= settings.redis.lock_ttl_seconds

        await store.release(KEY)
        assert await store.lock(KEY) is True

    asyncio.run(scenario())


def test_wait_for_result_returns_answer_stored_by_other_request(monkeypatch):
    monkeypatch.setattr(settings.redis, "lock_poll_seconds", 0.01)
    monkeypatch.setattr(settings.redis, "lock_wait_seconds", 1)

    async def scenario():
        store = new_store()
        response = make_response()

        async def other_request_finishes():
            await asyncio.sleep(0.05)
            await store.store(KEY, response)

        asyncio.create_task(other_request_finishes())
        assert await store.wait_for_result(KEY) == response

    asyncio.run(scenario())


def test_wait_for_result_gives_up_after_wait_time(monkeypatch):
    monkeypatch.setattr(settings.redis, "lock_poll_seconds", 0.01)
    monkeypatch.setattr(settings.redis, "lock_wait_seconds", 0.05)

    async def scenario():
        assert await new_store().wait_for_result(KEY) is None

    asyncio.run(scenario())


def test_other_content_type_does_not_hit_cache():
    async def scenario():
        store = new_store()
        await store.store(KEY, make_response())

        assert await store.get_cached(UploadKey(SHA, USER, "video", "post", "mv-test")) is None

    asyncio.run(scenario())


def test_other_context_does_not_hit_cache():
    async def scenario():
        store = new_store()
        await store.store(KEY, make_response())

        assert await store.get_cached(UploadKey(SHA, USER, "image", "breeding", "mv-test")) is None

    asyncio.run(scenario())


def test_redis_client_uses_timeouts_from_settings(monkeypatch):
    monkeypatch.setattr(settings.redis, "connect_timeout_seconds", 1.5)
    monkeypatch.setattr(settings.redis, "socket_timeout_seconds", 2.5)

    kwargs = make_redis().connection_pool.connection_kwargs

    assert kwargs["socket_connect_timeout"] == 1.5
    assert kwargs["socket_timeout"] == 2.5

