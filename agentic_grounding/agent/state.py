from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_grounding.schemas import CandidateScore, QuerySpec


@dataclass
class AgentState:
    query: QuerySpec
    ranked_candidates: list[CandidateScore]
    budget: int = 8
    evidence_log: list[dict[str, Any]] = field(default_factory=list)
    rejected_ids: set[str] = field(default_factory=set)
    selected_id: str | None = None
    stopped: bool = False

    @property
    def active_candidates(self) -> list[CandidateScore]:
        return [item for item in self.ranked_candidates if item.object_id not in self.rejected_ids]

    def record(self, action: str, observation: dict[str, Any]) -> None:
        self.evidence_log.append({"action": action, "observation": observation})
        self.budget -= 1

