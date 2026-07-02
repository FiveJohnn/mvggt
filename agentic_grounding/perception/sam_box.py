from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .base import ImageInput, Segmenter, load_pil


class SAMBoxSegmenter(Segmenter):
    """Classic Segment Anything box-prompt adapter for GroundingDINO boxes."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_type: str = "vit_h",
        device: str = "cuda",
    ) -> None:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "Install the official segment-anything package before using SAMBoxSegmenter"
            ) from exc
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        self.predictor = SamPredictor(sam.to(device).eval())
        self.device = device

    def segment_boxes(
        self,
        image: ImageInput,
        boxes_xyxy: np.ndarray,
    ) -> list[np.ndarray]:
        image_np = np.asarray(load_pil(image))
        if len(boxes_xyxy) == 0:
            return []
        self.predictor.set_image(image_np)
        boxes = torch.as_tensor(boxes_xyxy, dtype=torch.float32, device=self.device)
        transformed = self.predictor.transform.apply_boxes_torch(boxes, image_np.shape[:2])
        masks, scores, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=True,
        )
        best = scores.argmax(dim=1)
        selected = masks[torch.arange(len(best), device=best.device), best]
        return [mask.detach().cpu().numpy().astype(bool) for mask in selected]
