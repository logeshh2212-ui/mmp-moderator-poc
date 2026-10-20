import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


@pytest.fixture(autouse=True)
def temp_dir(tmp_path, monkeypatch):
    """Every test writes uploads to its own empty folder."""
    monkeypatch.setattr(settings, "temp_dir", tmp_path)
    return tmp_path


@pytest.fixture
def client():
    return TestClient(app)
