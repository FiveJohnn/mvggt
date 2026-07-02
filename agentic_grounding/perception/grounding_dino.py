from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from agentic_grounding.schemas import Detection2D

from .base import Detector, ImageInput, load_pil


class HuggingFaceGroundingDINO(Detector):
    """Grounding DINO adapter using the Transformers implementation.

    Pass a local model directory to avoid implicit network downloads.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        box_threshold: float = 0.20,
        text_threshold: float = 0.20,
    ) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path, local_files_only=True
        ).to(device).eval()
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def detect(
        self,
        image: ImageInput,
        categories: Sequence[str],
        view_id: int,
    ) -> list[Detection2D]:
        pil = load_pil(image)
        labels = [category.strip() for category in categories if category.strip()]
        if not labels:
            return []
        # The processor handles token/category alignment more reliably with a list.
        inputs = self.processor(images=pil, text=[labels], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([[pil.height, pil.width]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )[0]
        text_labels = results.get("text_labels", results.get("labels", []))
        detections: list[Detection2D] = []
        for index, (box, score, label) in enumerate(
            zip(results["boxes"], results["scores"], text_labels)
        ):
            if not isinstance(label, str):
                label = labels[int(label)] if int(label) < len(labels) else str(label)
            detections.append(
                Detection2D(
                    view_id=view_id,
                    detection_id=f"v{view_id:03d}_d{index:03d}",
                    category=label.strip().lower(),
                    score=float(score),
                    box_xyxy=box.detach().float().cpu().numpy().astype(np.float32),
                    phrase=label,
                )
            )
        return detections

