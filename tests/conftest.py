import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.cache.idempotency import IdempotencyStore, get_idempotency_store
from app.core.config import settings
from app.main import app
import shutil
import subprocess
from app.cache.blocklist import PhashBlocklist

@pytest.fixture(autouse=True)
def no_real_models(monkeypatch):
    """Normal tests never load SigLIP; real-model tests are marked 'models'."""
    from app.ml.registry import registry

    monkeypatch.setattr(settings.ml, "backbone", None)
    monkeypatch.setattr(settings.ml, "detector", None)
    monkeypatch.setattr(settings.ml, "species_model", None)
    monkeypatch.setattr(settings.ml, "nsfw_model", None)
    monkeypatch.setattr(settings.ml, "ai_model", None)
    monkeypatch.setattr(settings.ml, "ocr", False)
    monkeypatch.setattr(settings.ml, "audio_model", None)
    monkeypatch.setattr(settings.ml, "speech_model", None)
    yield
    registry.clear()


@pytest.fixture(autouse=True)
def temp_dir(tmp_path, monkeypatch):
    """Every test writes uploads (and any quarantined file) to its own empty folder."""
    monkeypatch.setattr(settings, "temp_dir", tmp_path)
    monkeypatch.setattr(settings.safety, "quarantine_dir", tmp_path / "quarantine")
    return tmp_path


@pytest.fixture
def redis_server():
    """One in-memory Redis per test, shared by the async and sync clients."""
    return fakeredis.FakeServer()

@pytest.fixture(autouse=True)
def blocklist(redis_server, monkeypatch):
    """Every test gets an empty in-memory blocklist instead of the real Redis."""
    fake = PhashBlocklist(fakeredis.FakeRedis(server=redis_server, decode_responses=True))
    monkeypatch.setattr("app.pipeline.orchestrator.get_blocklist", lambda: fake)
    return fake


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

def _ffmpeg(out, *args):
    subprocess.run(
        [shutil.which("ffmpeg"), "-y", "-v", "error", *args, str(out)],
        check=True,
        timeout=120,
    )
    return out


def _video_src(seconds):
    return ["-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=160x120:rate=10"]


def _audio_src(seconds):
    return ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]


H264 = ["-pix_fmt", "yuv420p", "-c:v", "libx264"]


@pytest.fixture(scope="session")
def videos(tmp_path_factory):
    """Generated test videos. Skips if ffmpeg/ffprobe are not installed."""
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not on PATH")

    folder = tmp_path_factory.mktemp("videos")
    v = {
        "mp4": _ffmpeg(folder / "ok.mp4", *_video_src(2), *H264),
        "mov": _ffmpeg(folder / "ok.mov", *_video_src(2), *H264),
        "avi": _ffmpeg(folder / "ok.avi", *_video_src(2), "-c:v", "mpeg4"),
        "with_audio": _ffmpeg(
            folder / "with_audio.mp4",
            *_video_src(2), *_audio_src(2), *H264, "-c:a", "aac", "-shortest",
        ),
        "60s": _ffmpeg(folder / "60s.mp4", *_video_src(60), *H264),
        "61s": _ffmpeg(folder / "61s.mp4", *_video_src(61), *H264),
        "audio_only": _ffmpeg(folder / "audio_only.mp4", *_audio_src(2), "-c:a", "aac"),
    }

    # ffmpeg writes the index (moov) at the end, so half a file cannot be read.
    data = v["mp4"].read_bytes()
    corrupt = folder / "corrupt.mp4"
    corrupt.write_bytes(data[: len(data) // 2])
    v["corrupt"] = corrupt

    return v

