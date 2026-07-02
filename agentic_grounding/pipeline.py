from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from agentic_grounding.fusion import AssociationConfig, ObjectRegistry, associate_observations, lift_observation
from agentic_grounding.geometry.base import GeometryBackend, GeometryResult
from agentic_grounding.perception.base import ConceptSegmenter, Detector, Segmenter
from agentic_grounding.perception.observations import observations_from_detections
from agentic_grounding.schemas import Detection2D, ObjectObservation, QuerySpec


@dataclass
class PipelineArtifacts:
    geometry: GeometryResult
    observations: list[ObjectObservation]
    registry: ObjectRegistry


class RegistryBuilder:
    def __init__(
        self,
        geometry_backend: GeometryBackend,
        detector: Detector | None = None,
        box_segmenter: Segmenter | None = None,
        concept_segmenter: ConceptSegmenter | None = None,
        association_config: AssociationConfig | None = None,
        geometry_confidence_threshold: float = 0.0,
    ) -> None:
        if concept_segmenter is None and (detector is None or box_segmenter is None):
            raise ValueError("Provide concept_segmenter or both detector and box_segmenter")
        self.geometry_backend = geometry_backend
        self.detector = detector
        self.box_segmenter = box_segmenter
        self.concept_segmenter = concept_segmenter
        self.association_config = association_config or AssociationConfig()
        self.geometry_confidence_threshold = geometry_confidence_threshold

    def build(
        self,
        image_paths: Sequence[str | Path | Image.Image],
        query: QuerySpec,
    ) -> PipelineArtifacts:
        geometry = self.geometry_backend.reconstruct(image_paths)
        observations: list[ObjectObservation] = []
        for view_id, image in enumerate(image_paths):
            if self.concept_segmenter is not None:
                for category in query.categories:
                    masks, boxes, scores = self.concept_segmenter.segment_concept(image, category)
                    detections = [
                        Detection2D(
                            view_id=view_id,
                            detection_id=f"v{view_id:03d}_{category}_{idx:03d}",
                            category=category,
                            score=float(scores[idx]),
                            box_xyxy=np.asarray(boxes[idx], dtype=np.float32),
                            phrase=category,
                        )
                        for idx in range(len(masks))
                    ]
                    observations.extend(observations_from_detections(detections, masks))
            else:
                assert self.detector is not None and self.box_segmenter is not None
                detections = self.detector.detect(image, query.categories, view_id)
                boxes = np.stack([item.box_xyxy for item in detections]) if detections else np.empty((0, 4))
                masks = self.box_segmenter.segment_boxes(image, boxes)
                observations.extend(observations_from_detections(detections, masks))
        for observation in observations:
            lift_observation(
                observation,
                geometry,
                confidence_threshold=self.geometry_confidence_threshold,
            )
        clusters = associate_observations(observations, self.association_config)
        registry = ObjectRegistry.from_clusters(clusters)
        return PipelineArtifacts(geometry=geometry, observations=observations, registry=registry)

