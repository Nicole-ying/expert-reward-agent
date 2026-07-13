from __future__ import annotations

import uuid
from typing import Any

from .schemas import EnvironmentContract, EvaluationEvidence, InterventionDecision, RewardCandidate


class RuleBasedExpertPlanner:
    """Deterministic reference planner used to test orchestration without an LLM."""

    def plan_initial(self, contract: EnvironmentContract, knowledge: list[dict[str, Any]]) -> InterventionDecision:
        return InterventionDecision(
            decision_id=_id(),
            level=0,
            target_component=None,
            transformation="compose_initial_reward",
            diagnosis="No trained policy evidence exists yet.",
            hypothesis="A reachable progress signal plus task completion evidence can establish learnability.",
            expected_changes={"score": "increase from random-policy baseline"},
            evidence_refs=[contract.environment_id, *[item.get("card_id", "card") for item in knowledge[:3]]],
            expert_assessment={
                "reachability": "Unknown before training; choose a dense signal available to early policies.",
                "directionality": "Use signed local improvement toward the declared task objective.",
                "scale": "Unknown before trajectories; start with bounded or physically normalized terms.",
                "credit_assignment": "Prefer step-level evidence because completion may be delayed.",
                "task_alignment": "Retain completion evidence so process progress is not the sole objective.",
                "exploit_resistance": "Check inaction, oscillation, and repeated-event incentives after evaluation.",
            },
            role_analysis={
                "process_progress": "required: supplies reachable directional feedback",
                "task_completion": "required when a declared completion signal is available; otherwise use a cautious proxy",
            },
            mathematical_plan={
                "progress": "state_improvement: provides signed local direction",
                "completion": "transition_event: avoids repeated occupancy reward when verified event evidence exists",
            },
        )

    def diagnose(
        self,
        contract: EnvironmentContract,
        current: RewardCandidate,
        evidence: EvaluationEvidence,
        history: list[dict[str, Any]],
    ) -> InterventionDecision:
        prior_interventions = sum(
            1 for item in history if (item.get("decision") or {}).get("level", 0) >= 1
        )
        if prior_interventions == 0:
            level = 1
            transformation = "repair_scale_or_sign"
            diagnosis = "The initial signal is reachable but its scale or sign may not dominate learning."
            hypothesis = "Repairing one component scale will improve score without changing reward semantics."
        elif prior_interventions == 1:
            level = 2
            transformation = "state_value_to_state_improvement"
            diagnosis = "Scale repair was insufficient; the current state-value proxy has weak credit assignment."
            hypothesis = "A state-improvement signal will provide directional step-level feedback."
        else:
            level = 3
            transformation = "rebuild_reward_structure"
            diagnosis = "The current reward family remains below target after local and structural repair."
            hypothesis = "A new component-role composition can escape the failed structural family."
        target = next(iter(current.component_roles), None)
        return InterventionDecision(
            decision_id=_id(),
            level=level,
            target_component=target,
            transformation=transformation,
            diagnosis=diagnosis,
            hypothesis=hypothesis,
            expected_changes={"mean_score": "increase", "exploit_flags": "not increase"},
            evidence_refs=[f"evaluation:{current.candidate_id}", f"score:{evidence.mean_score:.3f}"],
            expert_assessment={
                "reachability": "The progress component is active in evaluation.",
                "directionality": "External score remains the directional verification criterion.",
                "scale": "The first intervention checks whether signal magnitude is adequate.",
                "credit_assignment": "Escalate to state improvement when state occupancy gives weak action credit.",
                "task_alignment": "No exploit flag is observed, but target score is not yet reached.",
                "exploit_resistance": "Continue monitoring exploit flags during each intervention.",
            },
            role_analysis={"process_progress": "present but ineffective at the observed score"},
            mathematical_plan={
                "progress": "state_improvement: retain or introduce directional step-level credit"
            },
        )


def _id() -> str:
    return f"decision_{uuid.uuid4().hex[:8]}"
