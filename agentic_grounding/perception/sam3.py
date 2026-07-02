from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import ConceptSegmenter, ImageInput, load_pil


class SAM3ConceptSegmenter(ConceptSegmenter):
    """Adapter for the official SAM 3 image processor.

    SAM 3 currently requires a newer PyTorch/CUDA stack than the base MVGGT
    environment. Running it in a separate environment/service is supported by
    implementing the same ``ConceptSegmenter`` interface.
    """

    def __init__(self, model=None, processor=None) -> None:
        if model is None or processor is None:
            try:
                from sam3.model.sam3_image_processor import Sam3Processor
                from sam3.model_builder import build_sam3_image_model
            except ImportError as exc:
                raise ImportError(
                    "Install the official facebookresearch/sam3 package, preferably in a separate environment"
                ) from exc
            model = build_sam3_image_model()
            processor = Sam3Processor(model)
        self.model = model
        self.processor = processor

    def segment_concept(
        self,
        image: ImageInput,
        prompt: str,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        state = self.processor.set_image(load_pil(image))
        output = self.processor.set_text_prompt(state=state, prompt=prompt)
        masks_value = output["masks"]
        boxes_value = output["boxes"]
        scores_value = output["scores"]
        if hasattr(masks_value, "detach"):
            masks_value = masks_value.detach().cpu().numpy()
            boxes_value = boxes_value.detach().cpu().numpy()
            scores_value = scores_value.detach().cpu().numpy()
        masks = [np.asarray(mask).squeeze().astype(bool) for mask in masks_value]
        return masks, np.asarray(boxes_value, dtype=np.float32), np.asarray(scores_value, dtype=np.float32)

