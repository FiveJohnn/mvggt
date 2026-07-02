from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from agentic_grounding.schemas import Object3D, ObjectObservation


class ObjectRegistry:
    def __init__(self, objects: Iterable[Object3D] = ()) -> None:
        self.objects = {obj.object_id: obj for obj in objects}

    @classmethod
    def from_clusters(cls, clusters: list[list[ObjectObservation]]) -> "ObjectRegistry":
        objects: list[Object3D] = []
        for index, cluster in enumerate(clusters, start=1):
            point_groups = [item.points_world for item in cluster if item.points_world is not None]
            if not point_groups:
                continue
            points = np.concatenate(point_groups, axis=0)
            category_scores: dict[str, list[float]] = defaultdict(list)
            for item in cluster:
                category_scores[item.category].append(item.category_score)
            aggregated_scores = {
                label: float(np.mean(scores)) for label, scores in category_scores.items()
            }
            features = [item.appearance_feature for item in cluster if item.appearance_feature is not None]
            feature = np.mean(features, axis=0) if features else None
            objects.append(
                Object3D(
                    object_id=f"O{index:03d}",
                    category_scores=aggregated_scores,
                    observation_ids=[item.observation_id for item in cluster],
                    visible_views=sorted({item.view_id for item in cluster}),
                    points_world=points.astype(np.float32),
                    centroid=np.median(points, axis=0).astype(np.float32),
                    aabb_min=np.percentile(points, 1, axis=0).astype(np.float32),
                    aabb_max=np.percentile(points, 99, axis=0).astype(np.float32),
                    appearance_feature=None if feature is None else feature.astype(np.float32),
                    association_confidence=float(
                        np.mean([item.geometry_confidence for item in cluster])
                    ),
                )
            )
        return cls(objects)

    def by_category(self, category: str) -> list[Object3D]:
        normalized = category.strip().lower()
        return [obj for obj in self.objects.values() if obj.category.lower() == normalized]

    def get(self, object_id: str) -> Object3D:
        return self.objects[object_id]

    def __iter__(self):
        return iter(self.objects.values())

    def __len__(self) -> int:
        return len(self.objects)

    def to_summary(self) -> list[dict]:
        return [
            {
                "object_id": obj.object_id,
                "category": obj.category,
                "category_scores": obj.category_scores,
                "observation_ids": obj.observation_ids,
                "visible_views": obj.visible_views,
                "centroid": obj.centroid,
                "aabb_min": obj.aabb_min,
                "aabb_max": obj.aabb_max,
                "association_confidence": obj.association_confidence,
            }
            for obj in self.objects.values()
        ]

