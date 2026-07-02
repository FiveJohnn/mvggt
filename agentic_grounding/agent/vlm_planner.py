from __future__ import annotations

import json

from agentic_grounding.agent.state import AgentState
from agentic_grounding.vlm.base import VLMClient


PLANNER_SYSTEM_PROMPT = """You are a tool planner for zero-shot 3D referring segmentation.
The persistent Object IDs are the only valid entities. Never calculate coordinates or distances
yourself. Use MEASURE for geometry, LOOK for appearance, COMPARE for candidate diagnostics,
VERIFY before accepting a candidate, and STOP only after sufficient evidence.

Return one JSON object:
{"tool": "LOOK|MEASURE|COMPARE|VERIFY|STOP|ABORT", "arguments": {...}, "reason": "..."}
"""


class VLMToolPlanner:
    def __init__(self, client: VLMClient, max_history: int = 6) -> None:
        self.client = client
        self.max_history = max_history

    def next_action(self, state: AgentState) -> dict:
        candidates = [
            {
                "object_id": item.object_id,
                "total": item.total,
                "semantic": item.semantic,
                "relations": item.relations,
                "geometry": item.geometry,
                "conflicts": item.conflicts,
            }
            for item in state.active_candidates
        ]
        prompt = {
            "query": state.query.to_dict(),
            "budget_remaining": state.budget,
            "candidates": candidates,
            "recent_evidence": state.evidence_log[-self.max_history :],
            "tool_argument_examples": {
                "LOOK": {"object_ids": ["O001", "O002"]},
                "MEASURE": {
                    "subject_id": "O001", "relation": "left_of", "object_id": "O010",
                    "frame": "view_dependent", "reference_view": 3,
                },
                "COMPARE": {"candidate_ids": ["O001", "O002"]},
                "VERIFY": {"candidate_id": "O001"},
                "STOP": {"candidate_id": "O001", "confidence": 0.8},
            },
        }
        recent_evidence = state.evidence_log[-self.max_history :]
        images = [
            item["observation"]["image_path"]
            for item in recent_evidence
            if isinstance(item.get("observation"), dict)
            and item["observation"].get("image_path")
        ]
        action = self.client.complete_json(
            PLANNER_SYSTEM_PROMPT,
            json.dumps(prompt, ensure_ascii=False),
            images=images,
        )
        if "tool" not in action or "arguments" not in action:
            raise ValueError("VLM planner response must contain tool and arguments")
        return action
