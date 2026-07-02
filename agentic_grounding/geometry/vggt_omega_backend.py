from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .base import GeometryBackend, GeometryResult
from .common import invert_extrinsics, unproject_depth


class VGGTOmegaBackend(GeometryBackend):
    """Lazy adapter for the official ``facebookresearch/vggt-omega`` package."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        image_resolution: int = 512,
    ) -> None:
        try:
            from vggt_omega.models import VGGTOmega
            from vggt_omega.utils.load_fn import load_and_preprocess_images
            from vggt_omega.utils.pose_enc import encoding_to_camera
        except ImportError as exc:
            raise ImportError(
                "Install the official vggt-omega repository with `pip install -e .` first"
            ) from exc
        self._load_images = load_and_preprocess_images
        self._decode_camera = encoding_to_camera
        self.device = device
        self.image_resolution = image_resolution
        self.model = VGGTOmega().to(device).eval()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)

    def reconstruct(self, images: Sequence[str | Path | Image.Image]) -> GeometryResult:
        if any(isinstance(image, Image.Image) for image in images):
            raise TypeError("VGGTOmegaBackend currently expects image file paths")
        paths = [str(path) for path in images]
        batch = self._load_images(paths, image_resolution=self.image_resolution).to(self.device)
        with torch.inference_mode():
            predictions = self.model(batch)
        extrinsics, intrinsics = self._decode_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:]
        )
        depth = predictions["depth"]
        confidence = predictions.get("depth_conf", torch.ones_like(depth))
        depth_np = self._strip_batch_and_channel(depth)
        conf_np = self._strip_batch_and_channel(confidence)
        intrinsics_np = intrinsics.detach().float().cpu().numpy()
        extrinsics_np = extrinsics.detach().float().cpu().numpy()
        if intrinsics_np.ndim == 4:
            intrinsics_np = intrinsics_np[0]
            extrinsics_np = extrinsics_np[0]
        # Official camera decoding returns world-to-camera extrinsics.
        camera_to_world = invert_extrinsics(extrinsics_np)
        points = unproject_depth(depth_np, intrinsics_np, camera_to_world)
        result = GeometryResult(
            points_world=points,
            depth=depth_np,
            intrinsics=intrinsics_np,
            camera_to_world=camera_to_world,
            confidence=conf_np,
            image_hw=tuple(depth_np.shape[-2:]),
            metadata={"backend": "vggt-omega", "metric_scale": False},
        )
        result.validate()
        return result

    @staticmethod
    def _strip_batch_and_channel(tensor: torch.Tensor) -> np.ndarray:
        array = tensor.detach().float().cpu().numpy()
        if array.ndim >= 4 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim == 4:
            array = array[0]
        if array.ndim != 3:
            raise ValueError(f"Expected [V,H,W] prediction, got {array.shape}")
        return array.astype(np.float32, copy=False)
