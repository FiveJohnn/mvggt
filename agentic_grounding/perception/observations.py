from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from agentic_grounding.schemas import Detection2D, ObjectObservation


def observations_from_detections(
    detections: Sequence[Detection2D],
    masks: Sequence[np.ndarray],
) -> list[ObjectObservation]:
    if len(detections) != len(masks):
        raise ValueError("Detector and segmenter must return the same number of instances")
    return [
        ObjectObservation(
            observation_id=detection.detection_id,
            view_id=detection.view_id,
            category=detection.category,
            category_score=detection.score,
            box_xyxy=detection.box_xyxy,
            mask=np.asarray(mask).astype(bool),
        )
        for detection, mask in zip(detections, masks)
    ]

