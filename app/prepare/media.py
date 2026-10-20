from dataclasses import dataclass

from PIL import Image


@dataclass
class PreparedFrame:
    timestamp: float | None   # seconds into the video; None for a still image
    backbone: Image.Image     # long side = backbone_size; the model wrapper does the final transform
    detector: Image.Image     # long side = detector_size


@dataclass
class PreparedMedia:
    frames: list[PreparedFrame]  # one for an image, up to max_video_frames for a video
