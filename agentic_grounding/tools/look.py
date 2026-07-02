from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentic_grounding.fusion.object_registry import ObjectRegistry
from agentic_grounding.schemas import ObjectObservation
from agentic_grounding.visualization.candidate_cards import render_candidate_cards

from .base import Tool


class LookTool(Tool):
    name = "LOOK"
    description = "Render crops of selected persistent object IDs from their supporting views."

    def __init__(
        self,
        image_paths: Sequence[str | Path],
        registry: ObjectRegistry,
        observations: dict[str, ObjectObservation],
        output_dir: str | Path,
    ) -> None:
        self.image_paths = [str(path) for path in image_paths]
        self.registry = registry
        self.observations = observations
        self.output_dir = Path(output_dir)

    def __call__(self, object_ids: list[str], **_: Any) -> dict[str, Any]:
        candidates = []
        for object_id in object_ids:
            obj = self.registry.get(object_id)
            if not obj.observation_ids:
                continue
            observation = max(
                (self.observations[item] for item in obj.observation_ids),
                key=lambda item: item.category_score,
            )
            candidates.append(
                {"object_id": object_id, "view_id": observation.view_id, "box_xyxy": observation.box_xyxy}
            )
        path = self.output_dir / ("look_" + "_".join(object_ids) + ".jpg")
        render_candidate_cards(self.image_paths, candidates, path)
        return {"image_path": str(path), "candidates": candidates}

