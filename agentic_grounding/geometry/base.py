from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


@dataclass
class GeometryResult:
    points_world: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray | None
    camera_to_world: np.ndarray
    confidence: np.ndarray
    gravity: np.ndarray | None = None
    image_hw: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.points_world.ndim != 4 or self.points_world.shape[-1] != 3:
            raise ValueError("points_world must have shape [V,H,W,3]")
        if self.depth.shape != self.points_world.shape[:3]:
            raise ValueError("depth must have shape [V,H,W]")
        if self.confidence.shape != self.depth.shape:
            raise ValueError("confidence must have shape [V,H,W]")
        if self.camera_to_world.shape != (self.points_world.shape[0], 4, 4):
            raise ValueError("camera_to_world must have shape [V,4,4]")


class GeometryBackend(ABC):
    @abstractmethod
    def reconstruct(self, images: Sequence[str | Path | Image.Image]) -> GeometryResult:
        raise NotImplementedError


def save_geometry(path: str | Path, result: GeometryResult) -> None:
    result.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        points_world=result.points_world,
        depth=result.depth,
        intrinsics=np.array([]) if result.intrinsics is None else result.intrinsics,
        camera_to_world=result.camera_to_world,
        confidence=result.confidence,
        gravity=np.array([]) if result.gravity is None else result.gravity,
        image_hw=np.asarray(result.image_hw or result.depth.shape[1:]),
    )


def load_geometry(path: str | Path) -> GeometryResult:
    data = np.load(path, allow_pickle=False)
    intrinsics = data["intrinsics"]
    gravity = data["gravity"]
    result = GeometryResult(
        points_world=data["points_world"],
        depth=data["depth"],
        intrinsics=None if intrinsics.size == 0 else intrinsics,
        camera_to_world=data["camera_to_world"],
        confidence=data["confidence"],
        gravity=None if gravity.size == 0 else gravity,
        image_hw=tuple(int(v) for v in data["image_hw"]),
    )
    result.validate()
    return result

