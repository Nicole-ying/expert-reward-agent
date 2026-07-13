from __future__ import annotations

from dataclasses import dataclass

from .schemas import AgentState, EnvironmentContract, EvaluationEvidence, StopReason


@dataclass
class StopPolicy:
    max_scientific_iterations: int = 10
    max_stagnation: int = 4

    def after_evaluation(
        self,
        state: AgentState,
        contract: EnvironmentContract,
        evidence: EvaluationEvidence,
        confirmed: bool = False,
    ) -> StopReason | None:
        solved = evidence.mean_score >= contract.target_score and not evidence.exploit_flags
        if solved and confirmed:
            return StopReason.SOLVED_CONFIRMED
        if state.scientific_iteration >= self.max_scientific_iterations:
            return StopReason.BUDGET_EXHAUSTED
        if state.stagnation_count >= self.max_stagnation:
            return StopReason.SEARCH_STAGNATED
        return None
