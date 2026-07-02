from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .base import GeometryBackend, GeometryResult


def _load_rgb(image: str | Path | Image.Image, image_hw: tuple[int, int]) -> np.ndarray:
    pil = image if isinstance(image, Image.Image) else Image.open(image)
    height, width = image_hw
    return np.asarray(pil.convert("RGB").resize((width, height), Image.Resampling.BILINEAR))


class MVGGTBackend(GeometryBackend):
    """Geometry-only adapter around an already loaded MVGGT/Pi3 model.

    Instantiate the model with ``use_referring_segmentation=False``. This avoids
    loading or using the ScanRefer-trained referring head in a zero-shot system.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        image_hw: tuple[int, int] = (336, 518),
    ) -> None:
        self.model = model.eval().to(device)
        self.device = device
        self.image_hw = image_hw

    def reconstruct(self, images: Sequence[str | Path | Image.Image]) -> GeometryResult:
        arrays = [_load_rgb(image, self.image_hw) for image in images]
        tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model(tensor)
        points = output["points"][0].detach().float().cpu().numpy()
        local = output["local_points"][0].detach().float().cpu().numpy()
        poses = output["camera_poses"][0].detach().float().cpu().numpy()
        conf_tensor = output.get("conf")
        confidence = (
            np.ones(local.shape[:3], dtype=np.float32)
            if conf_tensor is None
            else torch.sigmoid(conf_tensor[0, ..., 0]).detach().float().cpu().numpy()
        )
        result = GeometryResult(
            points_world=points,
            depth=local[..., 2],
            intrinsics=None,
            camera_to_world=poses,
            confidence=confidence,
            image_hw=self.image_hw,
            metadata={"backend": "mvggt", "metric_scale": False},
        )
        result.validate()
        return result

