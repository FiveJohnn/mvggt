from __future__ import annotations

from typing import Any

import numpy as np

from agentic_grounding.fusion.object_registry import ObjectRegistry
from agentic_grounding.geometry.base import GeometryResult
from agentic_grounding.spatial.coordinate_frames import project_world_points, world_to_camera

from .base import Tool


class PropagateTool(Tool):
    name = "PROPAGATE"
    description = "Project a 3D object support into a target view and return visible point prompts."

    def __init__(self, registry: ObjectRegistry, geometry: GeometryResult) -> None:
        self.registry = registry
        self.geometry = geometry

    def __call__(self, object_id: str, target_view: int, max_prompts: int = 64, **_: Any) -> dict:
        obj = self.registry.get(object_id)
        points = obj.points_world
        if len(points) > max_prompts:
            indices = np.linspace(0, len(points) - 1, max_prompts).astype(int)
            points = points[indices]
        pose = self.geometry.camera_to_world[target_view]
        intrinsics = None if self.geometry.intrinsics is None else self.geometry.intrinsics[target_view]
        camera = world_to_camera(points, pose)
        pixels = project_world_points(points, pose, intrinsics)
        height, width = self.geometry.depth.shape[1:]
        valid = (
            (camera[:, 2] > 0)
            & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        )
        return {
            "object_id": object_id,
            "target_view": target_view,
            "positive_points_xy": pixels[valid].tolist(),
            "visible_fraction": float(valid.mean()) if len(valid) else 0.0,
        }

