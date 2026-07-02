from __future__ import annotations

import numpy as np
from PIL import Image

from agentic_grounding.geometry.base import GeometryResult
from agentic_grounding.schemas import ObjectObservation


def _resize_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if mask.shape == hw:
        return mask.astype(bool)
    height, width = hw
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (width, height), Image.Resampling.NEAREST
    )
    return np.asarray(resized) > 0


def lift_observation(
    observation: ObjectObservation,
    geometry: GeometryResult,
    confidence_threshold: float = 0.0,
    max_points: int = 5000,
    rng: np.random.Generator | None = None,
) -> ObjectObservation:
    view_id = observation.view_id
    mask = _resize_mask(observation.mask, geometry.depth.shape[1:])
    valid = mask & np.isfinite(geometry.depth[view_id]) & (geometry.depth[view_id] > 0)
    valid &= np.isfinite(geometry.points_world[view_id]).all(axis=-1)
    valid &= geometry.confidence[view_id] >= confidence_threshold
    points = geometry.points_world[view_id][valid]
    if len(points) > max_points:
        generator = rng or np.random.default_rng(0)
        points = points[generator.choice(len(points), max_points, replace=False)]
    observation.points_world = points.astype(np.float32)
    observation.geometry_confidence = (
        float(np.mean(geometry.confidence[view_id][valid])) if valid.any() else 0.0
    )
    observation.metadata["geometry_mask_shape"] = list(mask.shape)
    return observation

