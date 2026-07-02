from __future__ import annotations

from agentic_grounding.agent.state import AgentState


class DeterministicPlanner:
    """Transparent baseline planner before adding an LLM tool policy."""

    def __init__(self, stop_margin: float = 0.15) -> None:
        self.stop_margin = stop_margin

    def next_action(self, state: AgentState) -> dict:
        active = state.active_candidates
        if not active:
            return {"tool": "ABORT", "arguments": {"reason": "no candidates"}}
        if len(active) == 1:
            return {"tool": "VERIFY", "arguments": {"candidate_id": active[0].object_id}}
        if state.evidence_log and state.evidence_log[-1]["action"] == "COMPARE":
            return {"tool": "VERIFY", "arguments": {"candidate_id": active[0].object_id}}
        margin = active[0].total - active[1].total
        if margin >= self.stop_margin and not active[0].conflicts:
            return {"tool": "VERIFY", "arguments": {"candidate_id": active[0].object_id}}
        return {
            "tool": "COMPARE",
            "arguments": {"candidate_ids": [active[0].object_id, active[1].object_id]},
        }
