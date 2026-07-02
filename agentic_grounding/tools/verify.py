from __future__ import annotations

from typing import Any

from agentic_grounding.schemas import CandidateScore

from .base import Tool


class VerifyTool(Tool):
    name = "VERIFY"
    description = "Verify a ranked candidate against accumulated semantic and geometric evidence."

    def __init__(self, scores: dict[str, CandidateScore], pass_threshold: float = 0.45) -> None:
        self.scores = scores
        self.pass_threshold = pass_threshold

    def __call__(self, candidate_id: str, **_: Any) -> dict[str, Any]:
        candidate = self.scores[candidate_id]
        passed = candidate.total >= self.pass_threshold and not candidate.conflicts
        return {
            "candidate_id": candidate_id,
            "passed": passed,
            "score": candidate.total,
            "conflicts": candidate.conflicts,
            "evidence": candidate.evidence,
        }

