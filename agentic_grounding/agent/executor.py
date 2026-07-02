from __future__ import annotations

from typing import Any

from agentic_grounding.agent.state import AgentState
from agentic_grounding.tools.base import Tool


class AgentExecutor:
    def __init__(self, planner: Any, tools: dict[str, Tool]) -> None:
        self.planner = planner
        self.tools = tools

    def run(self, state: AgentState) -> AgentState:
        while state.budget > 0 and not state.stopped:
            action = self.planner.next_action(state)
            name = action["tool"]
            if name == "ABORT":
                state.record(name, action["arguments"])
                state.stopped = True
                break
            if name == "STOP":
                candidate_id = action["arguments"]["candidate_id"]
                if candidate_id not in {item.object_id for item in state.active_candidates}:
                    raise ValueError(f"Planner attempted to stop on inactive candidate {candidate_id}")
                state.record(name, {"candidate_id": candidate_id})
                state.selected_id = candidate_id
                state.stopped = True
                break
            if name not in self.tools:
                raise KeyError(f"Planner requested unavailable tool {name!r}")
            observation = self.tools[name](**action["arguments"])
            state.record(name, observation)
            if name == "VERIFY":
                candidate_id = action["arguments"]["candidate_id"]
                if observation["passed"]:
                    state.selected_id = candidate_id
                    state.stopped = True
                else:
                    state.rejected_ids.add(candidate_id)
            elif name == "COMPARE" and observation.get("margin") is not None:
                # Comparison itself is evidence; the next step verifies the current winner.
                winner = observation["winner"]
                state.ranked_candidates.sort(
                    key=lambda item: (item.object_id == winner, item.total), reverse=True
                )
        return state
