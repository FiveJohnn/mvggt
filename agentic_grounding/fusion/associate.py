from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentic_grounding.schemas import ObjectObservation


@dataclass
class AssociationConfig:
    voxel_size: float = 0.04
    voxel_dilation: int = 1
    min_pair_score: float = 0.38
    max_normalized_centroid_distance: float = 1.5
    weight_voxel: float = 0.55
    weight_appearance: float = 0.25
    weight_centroid: float = 0.20


def _normalize_feature(feature: np.ndarray | None) -> np.ndarray | None:
    if feature is None:
        return None
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 1e-8 else feature


def _voxel_set(points: np.ndarray, size: float, dilation: int = 0) -> set[tuple[int, int, int]]:
    if points is None or len(points) == 0:
        return set()
    voxels = np.floor(points / size).astype(np.int64)
    base = {tuple(value) for value in voxels}
    if dilation <= 0:
        return base
    expanded: set[tuple[int, int, int]] = set()
    for x, y, z in base:
        for dx in range(-dilation, dilation + 1):
            for dy in range(-dilation, dilation + 1):
                for dz in range(-dilation, dilation + 1):
                    expanded.add((x + dx, y + dy, z + dz))
    return expanded


def observation_pair_score(
    left: ObjectObservation,
    right: ObjectObservation,
    config: AssociationConfig,
) -> float:
    if left.view_id == right.view_id:
        return float("-inf")  # hard cannot-link
    if left.category != right.category:
        return float("-inf")
    if left.points_world is None or right.points_world is None:
        return float("-inf")
    left_voxels = _voxel_set(left.points_world, config.voxel_size, config.voxel_dilation)
    right_voxels = _voxel_set(right.points_world, config.voxel_size, config.voxel_dilation)
    if not left_voxels or not right_voxels:
        return float("-inf")
    intersection = len(left_voxels & right_voxels)
    voxel_score = intersection / max(1, min(len(left_voxels), len(right_voxels)))

    left_center = np.median(left.points_world, axis=0)
    right_center = np.median(right.points_world, axis=0)
    extent_left = np.linalg.norm(np.ptp(left.points_world, axis=0))
    extent_right = np.linalg.norm(np.ptp(right.points_world, axis=0))
    scale = max(0.5 * (extent_left + extent_right), config.voxel_size)
    normalized_distance = float(np.linalg.norm(left_center - right_center) / scale)
    if normalized_distance > config.max_normalized_centroid_distance:
        return float("-inf")
    centroid_score = max(0.0, 1.0 - normalized_distance / config.max_normalized_centroid_distance)

    feature_left = _normalize_feature(left.appearance_feature)
    feature_right = _normalize_feature(right.appearance_feature)
    appearance_score = (
        0.5
        if feature_left is None or feature_right is None
        else float(np.clip(np.dot(feature_left, feature_right), 0.0, 1.0))
    )
    return (
        config.weight_voxel * voxel_score
        + config.weight_appearance * appearance_score
        + config.weight_centroid * centroid_score
    )


def associate_observations(
    observations: list[ObjectObservation],
    config: AssociationConfig | None = None,
) -> list[list[ObjectObservation]]:
    """Greedy complete-link clustering with same-view cannot-link constraints."""
    cfg = config or AssociationConfig()
    clusters: list[list[ObjectObservation]] = []
    for observation in sorted(observations, key=lambda item: item.category_score, reverse=True):
        best_index: int | None = None
        best_score = float("-inf")
        for cluster_index, cluster in enumerate(clusters):
            pair_scores = [observation_pair_score(observation, member, cfg) for member in cluster]
            complete_link_score = min(pair_scores)
            if complete_link_score >= cfg.min_pair_score and complete_link_score > best_score:
                best_index = cluster_index
                best_score = complete_link_score
        if best_index is None:
            clusters.append([observation])
        else:
            clusters[best_index].append(observation)
    return clusters

