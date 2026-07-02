from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .base import GeometryBackend, GeometryResult
from .common import invert_extrinsics, unproject_depth


class VGGTBackend(GeometryBackend):
    """Adapter for the official ``facebookresearch/vggt`` implementation.

    ``checkpoint_path`` may be a downloaded Hugging Face model directory or a
    direct ``model.pt`` state-dict file. Nothing is downloaded by this class.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
    ) -> None:
        try:
            from vggt.models.vggt import VGGT
            from vggt.utils.load_fn import load_and_preprocess_images
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        except ImportError as exc:
            raise ImportError(
                "Install the official facebookresearch/vggt repository with "
                "`pip install -e /path/to/vggt` first"
            ) from exc

        checkpoint = Path(checkpoint_path)
        if checkpoint.is_file():
            model = VGGT()
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        elif checkpoint.is_dir():
            model = VGGT.from_pretrained(str(checkpoint), local_files_only=True)
        else:
            raise FileNotFoundError(f"VGGT checkpoint not found: {checkpoint}")

        self.model = model.to(device).eval()
        self.device = device
        self.dtype = dtype or (
            torch.bfloat16
            if device.startswith("cuda") and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        self._load_images = load_and_preprocess_images
        self._decode_camera = pose_encoding_to_extri_intri

    def reconstruct(self, images: Sequence[str | Path | Image.Image]) -> GeometryResult:
        if any(isinstance(image, Image.Image) for image in images):
            raise TypeError("VGGTBackend currently expects image file paths")
        batch = self._load_images([str(path) for path in images]).to(self.device)
        autocast_enabled = self.device.startswith("cuda")
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=self.dtype, enabled=autocast_enabled
        ):
            predictions = self.model(batch)

        depth = predictions["depth"]
        confidence = predictions.get("depth_conf", torch.ones_like(depth))
        extrinsics, intrinsics = self._decode_camera(
            predictions["pose_enc"], batch.shape[-2:]
        )
        depth_np = self._strip_batch_and_channel(depth)
        confidence_np = self._strip_batch_and_channel(confidence)
        intrinsics_np = self._strip_batch(intrinsics)
        extrinsics_np = self._strip_batch(extrinsics)
        camera_to_world = invert_extrinsics(extrinsics_np)
        points = unproject_depth(depth_np, intrinsics_np, camera_to_world)
        result = GeometryResult(
            points_world=points,
            depth=depth_np,
            intrinsics=intrinsics_np,
            camera_to_world=camera_to_world,
            confidence=confidence_np,
            image_hw=tuple(depth_np.shape[-2:]),
            metadata={"backend": "vggt", "metric_scale": False},
        )
        result.validate()
        return result

    @staticmethod
    def _strip_batch(tensor: torch.Tensor) -> np.ndarray:
        array = tensor.detach().float().cpu().numpy()
        return array[0] if array.ndim == 4 else array

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
