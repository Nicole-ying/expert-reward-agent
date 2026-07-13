from __future__ import annotations

from typing import Any, Protocol

from ..agent.schemas import (
    EnvironmentContract,
    EvaluationEvidence,
    InterventionDecision,
    RewardCandidate,
)


class ExecutionBackend(Protocol):
    def perceive(self) -> dict[str, Any]: ...

    def build_contract(self, perception: dict[str, Any]) -> EnvironmentContract: ...

    def generate_candidate(
        self,
        contract: EnvironmentContract,
        knowledge: list[dict[str, Any]],
        parent: RewardCandidate | None,
        decision: InterventionDecision | None,
    ) -> RewardCandidate: ...

    def validate_candidate(self, candidate: RewardCandidate) -> dict[str, Any]: ...

    def train(self, candidate: RewardCandidate, scientific_iteration: int) -> dict[str, Any]: ...

    def evaluate(self, candidate: RewardCandidate, training_artifact: dict[str, Any]) -> EvaluationEvidence: ...


class ExpertPlanner(Protocol):
    def plan_initial(self, contract: EnvironmentContract, knowledge: list[dict[str, Any]]) -> InterventionDecision: ...

    def diagnose(
        self,
        contract: EnvironmentContract,
        current: RewardCandidate,
        evidence: EvaluationEvidence,
        history: list[dict[str, Any]],
    ) -> InterventionDecision: ...
