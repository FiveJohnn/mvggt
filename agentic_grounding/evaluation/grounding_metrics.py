from __future__ import annotations

import numpy as np


def binary_iou(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    return float((intersection + eps) / (union + eps))

