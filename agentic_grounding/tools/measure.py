from __future__ import annotations

from typing import Any

from agentic_grounding.fusion.object_registry import ObjectRegistry
from agentic_grounding.spatial.relations import SpatialRelationEngine

from .base import Tool


class MeasureTool(Tool):
    name = "MEASURE"
    description = "Compute a deterministic geometric relation between two persistent 3D object IDs."

    def __init__(self, registry: ObjectRegistry, engine: SpatialRelationEngine) -> None:
        self.registry = registry
        self.engine = engine

    def __call__(
        self,
        subject_id: str,
        relation: str,
        object_id: str,
        frame: str = "world",
        reference_view: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        result = self.engine.measure(
            self.registry.get(subject_id),
            relation,
            self.registry.get(object_id),
            frame=frame,
            reference_view=reference_view,
        )
        return result.to_dict()

