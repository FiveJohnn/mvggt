from __future__ import annotations

from typing import Any

from agentic_grounding.schemas import CandidateScore

from .base import Tool


class CompareTool(Tool):
    name = "COMPARE"
    description = "Compare candidate scores and return the most discriminative unresolved components."

    def __init__(self, scores: dict[str, CandidateScore]) -> None:
        self.scores = scores

    def __call__(self, candidate_ids: list[str], **_: Any) -> dict[str, Any]:
        candidates = [self.scores[item] for item in candidate_ids]
        candidates.sort(key=lambda item: item.total, reverse=True)
        winner = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None
        return {
            "winner": winner.object_id,
            "runner_up": None if runner_up is None else runner_up.object_id,
            "margin": None if runner_up is None else winner.total - runner_up.total,
            "scores": {
                item.object_id: {
                    "total": item.total,
                    "semantic": item.semantic,
                    "relations": item.relations,
                    "geometry": item.geometry,
                    "conflicts": item.conflicts,
                }
                for item in candidates
            },
        }

