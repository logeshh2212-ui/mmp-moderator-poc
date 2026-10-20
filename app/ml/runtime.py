import logging

import app.core.config  # noqa: F401  must come before torch / transformers
import torch

from app.core.config import settings


logger = logging.getLogger("mmp.ml")


def configure_torch() -> str:
    """Apply thread and device settings once. Returns the device actually used."""
    torch.set_num_threads(settings.ml.num_threads)

    device = settings.ml.device
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ml.device is 'cuda' but no NVIDIA GPU is available")

    logger.info(
        "torch %s on %s, %d threads", torch.__version__, device, torch.get_num_threads()
    )
    return device
