from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from agentic_grounding.schemas import Detection2D


ImageInput = str | Path | Image.Image


class Detector(ABC):
    @abstractmethod
    def detect(
        self,
        image: ImageInput,
        categories: Sequence[str],
        view_id: int,
    ) -> list[Detection2D]:
        raise NotImplementedError


class Segmenter(ABC):
    @abstractmethod
    def segment_boxes(
        self,
        image: ImageInput,
        boxes_xyxy: np.ndarray,
    ) -> list[np.ndarray]:
        raise NotImplementedError


class ConceptSegmenter(ABC):
    @abstractmethod
    def segment_concept(
        self,
        image: ImageInput,
        prompt: str,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """Return masks, xyxy boxes, and scores for every matching instance."""
        raise NotImplementedError


def load_pil(image: ImageInput) -> Image.Image:
    return image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")

