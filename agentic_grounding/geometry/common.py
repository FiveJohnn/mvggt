from __future__ import annotations

import numpy as np


def invert_extrinsics(world_to_camera: np.ndarray) -> np.ndarray:
    """Convert batched OpenCV world-to-camera matrices to camera-to-world 4x4s."""
    if world_to_camera.ndim != 3 or world_to_camera.shape[-2:] not in {(3, 4), (4, 4)}:
        raise ValueError(
            "Extrinsics must have shape [V,3,4] or [V,4,4], "
            f"got {world_to_camera.shape}"
        )
    if world_to_camera.shape[-2:] == (3, 4):
        homogeneous = np.broadcast_to(
            np.eye(4, dtype=world_to_camera.dtype),
            (world_to_camera.shape[0], 4, 4),
        ).copy()
        homogeneous[:, :3, :] = world_to_camera
        world_to_camera = homogeneous
    return np.linalg.inv(world_to_camera)


def unproject_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    """Unproject [V,H,W] depth maps into a shared world coordinate system."""
    views, height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1).reshape(-1, 3)
    output = np.empty((views, height, width, 3), dtype=np.float32)
    for view_id in range(views):
        rays = pixels @ np.linalg.inv(intrinsics[view_id]).T
        camera_points = rays * depth[view_id].reshape(-1, 1)
        homogeneous = np.concatenate(
            [camera_points, np.ones((camera_points.shape[0], 1), dtype=camera_points.dtype)],
            axis=1,
        )
        world = homogeneous @ camera_to_world[view_id].T
        output[view_id] = world[:, :3].reshape(height, width, 3)
    return output


def resize_intrinsics(
    intrinsics: np.ndarray,
    old_hw: tuple[int, int],
    new_hw: tuple[int, int],
) -> np.ndarray:
    old_h, old_w = old_hw
    new_h, new_w = new_hw
    scaled = intrinsics.copy()
    scaled[..., 0, :] *= new_w / old_w
    scaled[..., 1, :] *= new_h / old_h
    scaled[..., 2, 2] = 1.0
    return scaled
