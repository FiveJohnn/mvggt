from __future__ import annotations

import numpy as np

from agentic_grounding.geometry.base import GeometryResult
from agentic_grounding.schemas import Object3D, RelationMeasurement

from .coordinate_frames import project_world_points, world_to_camera


def _surface_distance(left: Object3D, right: Object3D, max_points: int = 512) -> float:
    left_points = left.points_world[:: max(1, len(left.points_world) // max_points)][:max_points]
    right_points = right.points_world[:: max(1, len(right.points_world) // max_points)][:max_points]
    differences = left_points[:, None, :] - right_points[None, :, :]
    distances = np.linalg.norm(differences, axis=-1)
    nearest = np.concatenate([distances.min(axis=0), distances.min(axis=1)])
    return float(np.percentile(nearest, 5))


class SpatialRelationEngine:
    def __init__(self, geometry: GeometryResult | None = None) -> None:
        self.geometry = geometry

    def measure(
        self,
        subject: Object3D,
        predicate: str,
        obj: Object3D,
        frame: str = "world",
        reference_view: int | None = None,
    ) -> RelationMeasurement:
        op = predicate.lower().strip()
        if op in {"near", "closest_to", "farthest_from", "next_to"}:
            return self._distance_relation(subject, op, obj, frame)
        if op in {"above", "below", "on", "under"}:
            return self._gravity_relation(subject, op, obj)
        if op in {"left_of", "right_of", "front_of", "behind"}:
            return self._direction_relation(subject, op, obj, frame, reference_view)
        if op in {"inside", "contains", "touching"}:
            return self._topology_relation(subject, op, obj)
        return RelationMeasurement(
            subject.object_id, op, obj.object_id, frame, None, 0.0,
            {"error": f"unsupported predicate: {op}"}, confidence=0.0,
        )

    def _distance_relation(
        self, subject: Object3D, op: str, obj: Object3D, frame: str
    ) -> RelationMeasurement:
        center = float(np.linalg.norm(subject.centroid - obj.centroid))
        surface = _surface_distance(subject, obj)
        scale = max(np.sqrt(max(subject.diagonal * obj.diagonal, 1e-8)), 1e-6)
        normalized = surface / scale
        if op == "next_to":
            satisfied = normalized < 0.35
            score = float(np.exp(-normalized / 0.35))
        else:
            satisfied = None
            score = 1.0 / (1.0 + normalized)
            if op == "farthest_from":
                score = 1.0 - score
        return RelationMeasurement(
            subject.object_id, op, obj.object_id, frame, satisfied, score,
            {"center_distance": center, "surface_distance": surface, "normalized_distance": normalized},
        )

    def _gravity_relation(self, subject: Object3D, op: str, obj: Object3D) -> RelationMeasurement:
        gravity = None if self.geometry is None else self.geometry.gravity
        confidence = 1.0
        if gravity is None:
            # Fallback assumes a z-up scene, so gravity points toward -z.
            gravity = np.array([0.0, 0.0, -1.0])
            confidence = 0.35
        up = -np.asarray(gravity, dtype=np.float64)
        up /= max(np.linalg.norm(up), 1e-8)
        subject_height = float(subject.centroid @ up)
        object_height = float(obj.centroid @ up)
        margin = subject_height - object_height
        scale = max(0.5 * (subject.diagonal + obj.diagonal), 1e-6)
        normalized = margin / scale
        satisfied = normalized > 0.12 if op in {"above", "on"} else normalized < -0.12
        score = float(1.0 / (1.0 + np.exp(-abs(normalized) * 5.0)))
        return RelationMeasurement(
            subject.object_id, op, obj.object_id, "gravity", satisfied, score,
            {"height_margin": margin, "normalized_margin": normalized}, confidence=confidence,
        )

    def _direction_relation(
        self,
        subject: Object3D,
        op: str,
        obj: Object3D,
        frame: str,
        reference_view: int | None,
    ) -> RelationMeasurement:
        if self.geometry is None or reference_view is None:
            return RelationMeasurement(
                subject.object_id, op, obj.object_id, frame, None, 0.0,
                {"error": "directional relation requires geometry and reference_view"}, confidence=0.0,
            )
        pose = self.geometry.camera_to_world[reference_view]
        intrinsics = None if self.geometry.intrinsics is None else self.geometry.intrinsics[reference_view]
        centers = np.stack([subject.centroid, obj.centroid])
        camera_centers = world_to_camera(centers, pose)
        pixels = project_world_points(centers, pose, intrinsics)
        if op in {"left_of", "right_of"}:
            margin = float(pixels[1, 0] - pixels[0, 0])
            satisfied = margin > 0 if op == "left_of" else margin < 0
            axis = "image_u"
        else:
            margin = float(camera_centers[1, 2] - camera_centers[0, 2])
            satisfied = margin > 0 if op == "front_of" else margin < 0
            axis = "camera_z"
        score = float(1.0 - np.exp(-abs(margin) / (abs(margin) + 1.0)))
        return RelationMeasurement(
            subject.object_id, op, obj.object_id, f"view_{reference_view}", satisfied, score,
            {"signed_margin": margin, "axis": axis}, confidence=0.9,
            supporting_views=[reference_view],
        )

    def _topology_relation(self, subject: Object3D, op: str, obj: Object3D) -> RelationMeasurement:
        subject_inside = bool(
            np.all(subject.aabb_min >= obj.aabb_min) and np.all(subject.aabb_max <= obj.aabb_max)
        )
        object_inside = bool(
            np.all(obj.aabb_min >= subject.aabb_min) and np.all(obj.aabb_max <= subject.aabb_max)
        )
        gap = _surface_distance(subject, obj)
        if op == "inside":
            satisfied = subject_inside
        elif op == "contains":
            satisfied = object_inside
        else:
            satisfied = gap < 0.05 * max(subject.diagonal, obj.diagonal, 1e-6)
        return RelationMeasurement(
            subject.object_id, op, obj.object_id, "world", satisfied, float(satisfied),
            {"surface_distance": gap, "subject_inside_object": subject_inside, "object_inside_subject": object_inside},
        )
