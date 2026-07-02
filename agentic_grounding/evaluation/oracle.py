from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .grounding_metrics import binary_iou


def oracle_at_k(
    candidate_masks: Sequence[np.ndarray],
    target_mask: np.ndarray,
    ks: Sequence[int] = (1, 2, 4, 8),
) -> dict[int, float]:
    scores = [binary_iou(mask, target_mask) for mask in candidate_masks]
    return {int(k): max(scores[: int(k)], default=0.0) for k in ks}

