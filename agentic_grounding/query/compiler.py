from __future__ import annotations

from typing import Any

from agentic_grounding.schemas import QuerySpec
from agentic_grounding.vlm.base import VLMClient


SYSTEM_PROMPT = """You are a 3D referring-expression query compiler.
Convert the user expression into strict JSON. Do not solve the grounding task.
Identify the target, landmark/anchor objects, visual attributes, spatial
predicates, and the coordinate frame required by every predicate.

Allowed relation frames:
- gravity: above, below, on, under
- world: near, closest_to, farthest_from, between, inside, touching
- view_dependent: left, right, front, behind as seen from a camera/viewer
- object_centric: facing or relative to an object's intrinsic front
- unknown: insufficient evidence; never guess a frame

Return exactly this schema:
{
  "target": {"category": "...", "attributes": ["..."]},
  "anchors": [{"anchor_id": "a0", "category": "...", "attributes": []}],
  "predicates": [{
    "op": "left_of", "subject": "target", "object": "a0",
    "frame": "view_dependent", "confidence": 0.0,
    "hard": true, "reference_view": null
  }],
  "hard_distractor_policy": {
    "same_category": true, "similar_attributes": true
  }
}
Use short singular English category names even if the query is in another language.
"""


class QueryCompiler:
    def __init__(self, client: VLMClient) -> None:
        self.client = client

    def compile(self, text: str, query_id: str = "") -> QuerySpec:
        payload = self.client.complete_json(SYSTEM_PROMPT, f"Expression: {text}")
        payload["query_id"] = query_id
        payload["raw_text"] = text
        spec = QuerySpec.from_dict(payload)
        self._validate(spec)
        return spec

    @staticmethod
    def _validate(spec: QuerySpec) -> None:
        if not spec.target.category.strip():
            raise ValueError("Compiled target category is empty")
        anchor_ids = {anchor.anchor_id for anchor in spec.anchors}
        if len(anchor_ids) != len(spec.anchors):
            raise ValueError("Compiled anchor IDs must be unique")
        for predicate in spec.predicates:
            if predicate.object and predicate.object not in anchor_ids:
                raise ValueError(f"Predicate references unknown anchor {predicate.object!r}")
            if not 0.0 <= predicate.confidence <= 1.0:
                raise ValueError("Predicate confidence must be in [0, 1]")

    def compile_record(
        self,
        record: dict[str, Any],
        query_key: str = "description",
        id_key: str = "query_id",
    ) -> dict[str, Any]:
        raw = record[query_key]
        text = " ".join(raw) if isinstance(raw, list) else str(raw)
        query_id = str(record.get(id_key, record.get("ann_id", "")))
        output = dict(record)
        output["compiled_query"] = self.compile(text, query_id).to_dict()
        return output

