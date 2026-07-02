from __future__ import annotations

import numpy as np


def world_to_camera(points_world: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    world_to_cam = np.linalg.inv(camera_to_world)
    points = np.atleast_2d(points_world)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (homogeneous @ world_to_cam.T)[:, :3]


def project_world_points(
    points_world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: np.ndarray | None,
) -> np.ndarray:
    points_camera = world_to_camera(points_world, camera_to_world)
    if intrinsics is None:
        return points_camera[:, :2] / np.clip(points_camera[:, 2:3], 1e-6, None)
    pixels = points_camera @ intrinsics.T
    return pixels[:, :2] / np.clip(pixels[:, 2:3], 1e-6, None)

