import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.cache.idempotency import IdempotencyStore, get_idempotency_store
from app.core.config import settings
from app.main import app


@pytest.fixture(autouse=True)
def temp_dir(tmp_path, monkeypatch):
    """Every test writes uploads to its own empty folder."""
    monkeypatch.setattr(settings, "temp_dir", tmp_path)
    return tmp_path


@pytest.fixture
def redis_server():
    """One in-memory Redis per test, shared by the async and sync clients."""
    return fakeredis.FakeServer()


@pytest.fixture
def sync_redis(redis_server):
    """Lets tests look inside Redis without an event loop."""
    return fakeredis.FakeRedis(server=redis_server)


@pytest.fixture
def client(redis_server):
    store = IdempotencyStore(fakeredis.FakeAsyncRedis(server=redis_server))
    app.dependency_overrides[get_idempotency_store] = lambda: store
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
