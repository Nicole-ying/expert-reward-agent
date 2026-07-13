from __future__ import annotations

from typing import Any

from ..agent.schemas import (
    ComponentEvidence,
    EnvironmentContract,
    EvaluationEvidence,
    InterventionDecision,
    RewardCandidate,
)


STATE_VALUE_TEMPLATE = '''def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    progress = -abs(float(next_obs[0])) * {scale}
    return progress, {{"progress": progress}}
'''

STATE_IMPROVEMENT_TEMPLATE = '''def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    progress = (abs(float(obs[0])) - abs(float(next_obs[0]))) * {scale}
    return progress, {{"progress": progress}}
'''


class MockBackend:
    """Deterministic backend proving the complete control flow without RL cost."""

    def __init__(self):
        self.generated = 0

    def perceive(self) -> dict[str, Any]:
        return {
            "environment_id": "MockNavigation-v0",
            "task_description": "Move a scalar state toward zero and settle.",
            "observation_fields": [{"index": 0, "name": "distance"}],
            "action_description": {"type": "Discrete", "n": 3},
            "termination_conditions": ["goal", "timeout"],
            "target_score": 200.0,
            "max_episode_steps": 500,
            "available_signals": ["distance", "termination"],
        }

    def build_contract(self, perception: dict[str, Any]) -> EnvironmentContract:
        return EnvironmentContract(**perception)

    def generate_candidate(
        self,
        contract: EnvironmentContract,
        knowledge: list[dict[str, Any]],
        parent: RewardCandidate | None,
        decision: InterventionDecision | None,
    ) -> RewardCandidate:
        self.generated += 1
        scales = [0.01, 1.0, 20.0, 10.0]
        scale = scales[min(self.generated - 1, len(scales) - 1)]
        return RewardCandidate(
            candidate_id=f"candidate_{self.generated:02d}",
            code=(STATE_IMPROVEMENT_TEMPLATE if self.generated >= 3 else STATE_VALUE_TEMPLATE).format(scale=scale),
            component_roles={"progress": "process_progress"},
            mathematical_forms={"progress": "state_improvement" if self.generated >= 3 else "state_value"},
            parent_id=parent.candidate_id if parent else None,
            intervention_id=decision.decision_id if decision else None,
        )

    def validate_candidate(self, candidate: RewardCandidate) -> dict[str, Any]:
        compile(candidate.code, f"<{candidate.candidate_id}>", "exec")
        return {"valid": True}

    def train(self, candidate: RewardCandidate, scientific_iteration: int) -> dict[str, Any]:
        return {"model_id": f"model_{candidate.candidate_id}", "iteration": scientific_iteration}

    def evaluate(self, candidate: RewardCandidate, training_artifact: dict[str, Any]) -> EvaluationEvidence:
        scores = {"candidate_01": -80.0, "candidate_02": 35.0, "candidate_03": 225.0}
        score = scores.get(candidate.candidate_id, 218.0)
        return EvaluationEvidence(
            candidate_id=candidate.candidate_id,
            mean_score=score,
            score_std=5.0,
            mean_length=220.0,
            episodes=20,
            termination_counts={"goal": 18 if score >= 200 else 1, "timeout": 2 if score >= 200 else 19},
            behavior_metrics={"final_distance_mean": 0.05 if score >= 200 else 1.2},
            components=[ComponentEvidence("progress", "process_progress", score / 220.0, abs(score / 220.0), score, 0.9, score / 200.0)],
        )
