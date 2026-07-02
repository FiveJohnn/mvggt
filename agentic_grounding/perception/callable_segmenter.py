from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .base import ImageInput, Segmenter


class CallableBoxSegmenter(Segmenter):
    """Wrap SAM/SAM2/SAM3 box-prompt code without coupling to its package."""

    def __init__(self, fn: Callable[[ImageInput, np.ndarray], list[np.ndarray]]) -> None:
        self.fn = fn

    def segment_boxes(self, image: ImageInput, boxes_xyxy: np.ndarray) -> list[np.ndarray]:
        return [np.asarray(mask).astype(bool) for mask in self.fn(image, boxes_xyxy)]

