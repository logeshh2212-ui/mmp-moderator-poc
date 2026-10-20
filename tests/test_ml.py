import os

import pytest

from app.ml.registry import ModelRegistry


class FakeModel:
    def __init__(self, name="fake", fail=False):
        self.name = name
        self.fail = fail
        self.loads = 0
        self.warmups = 0

    def load(self):
        if self.fail:
            raise OSError("weights not found")
        self.loads += 1

    def warmup(self):
        self.warmups += 1


def test_model_is_loaded_and_warmed_up_once():
    registry = ModelRegistry()
    model = FakeModel()
    registry.register(model)

    registry.load_all()

    assert registry.status == {"fake": "ready"}
    assert (model.loads, model.warmups) == (1, 1)
    assert registry.get("fake") is model


def test_failed_model_does_not_stop_the_others():
    registry = ModelRegistry()
    registry.register(FakeModel("broken", fail=True))
    registry.register(FakeModel("good"))

    registry.load_all()

    assert registry.status["good"] == "ready"
    assert registry.status["broken"].startswith("failed: weights not found")
    with pytest.raises(RuntimeError, match="not ready"):
        registry.get("broken")


def test_unknown_model_is_an_error():
    with pytest.raises(RuntimeError, match="unknown"):
        ModelRegistry().get("nope")


def test_torch_is_the_cpu_build():
    import torch

    assert torch.cuda.is_available() is False
    assert torch.get_num_threads() >= 1


@pytest.mark.skipif(not os.environ.get("HF_HOME"), reason=".env not loaded")
def test_model_cache_is_on_d():
    import app.ml  # noqa: F401  loads .env
    from huggingface_hub import constants

    assert constants.HF_HUB_CACHE.upper().startswith("D:")
