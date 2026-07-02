from __future__ import annotations

import base64
import io
import json
import urllib.request

import numpy as np
from PIL import Image

from .base import ConceptSegmenter, ImageInput, load_pil


class HTTPConceptSegmenter(ConceptSegmenter):
    """Client for running SAM 3 in an isolated, version-compatible environment."""

    def __init__(self, endpoint: str = "http://127.0.0.1:8765/segment", timeout: float = 180.0) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    def segment_concept(
        self, image: ImageInput, prompt: str
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        buffer = io.BytesIO()
        load_pil(image).save(buffer, format="JPEG", quality=95)
        payload = {
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "prompt": prompt,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        masks = []
        for encoded in result["masks_png_base64"]:
            mask = Image.open(io.BytesIO(base64.b64decode(encoded)))
            masks.append(np.asarray(mask) > 0)
        return masks, np.asarray(result["boxes_xyxy"], dtype=np.float32), np.asarray(
            result["scores"], dtype=np.float32
        )

