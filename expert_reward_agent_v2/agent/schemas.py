from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    INIT = "init"
    PERCEIVE = "perceive"
    CONTRACT = "contract"
    PLAN = "plan"
    GENERATE = "generate"
    VALIDATE = "validate"
    TRAIN = "train"
    EVALUATE = "evaluate"
    DIAGNOSE = "diagnose"
    INTERVENE = "intervene"
    CONFIRM = "confirm"
    STOPPED = "stopped"
    FAILED = "failed"


class Action(str, Enum):
    PERCEIVE = "perceive"
    BUILD_CONTRACT = "build_contract"
    RETRIEVE_KNOWLEDGE = "retrieve_knowledge"
    PLAN_INITIAL_REWARD = "plan_initial_reward"
    GENERATE_CODE = "generate_code"
    VALIDATE_CODE = "validate_code"
    TRAIN = "train"
    EVALUATE = "evaluate"
    DIAGNOSE = "diagnose"
    REVISE_LEVEL_1 = "revise_level_1"
    REVISE_LEVEL_2 = "revise_level_2"
    REBUILD_LEVEL_3 = "rebuild_level_3"
    CONFIRM_BEST = "confirm_best"
    STOP = "stop"


class StopReason(str, Enum):
    SOLVED_CONFIRMED = "solved_confirmed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    SEARCH_STAGNATED = "search_stagnated"
    ENVIRONMENT_UNSUPPORTED = "environment_unsupported"
    ENGINEERING_FAILURE = "engineering_failure"


class EvidenceKind(str, Enum):
    GROUND_TRUTH = "ground_truth"
    PROXY = "proxy"
    UNKNOWN = "unknown"


@dataclass
class EnvironmentContract:
    environment_id: str
    task_description: str
    observation_fields: list[dict[str, Any]]
    action_description: dict[str, Any]
    termination_conditions: list[str]
    target_score: float
    max_episode_steps: int
    available_signals: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)


@dataclass
class RewardCandidate:
    candidate_id: str
    code: str
    component_roles: dict[str, str]
    mathematical_forms: dict[str, str]
    parent_id: str | None = None
    intervention_id: str | None = None


@dataclass
class ComponentEvidence:
    name: str
    role: str
    mean: float
    abs_mean: float
    episode_sum_mean: float
    active_rate: float
    mean_when_active: float


@dataclass
class EvaluationEvidence:
    candidate_id: str
    mean_score: float
    score_std: float
    mean_length: float
    episodes: int
    termination_counts: dict[str, int]
    behavior_metrics: dict[str, float | int | str | None]
    components: list[ComponentEvidence] = field(default_factory=list)
    exploit_flags: list[str] = field(default_factory=list)
    source: str = "final_policy_evaluation"


@dataclass
class InterventionDecision:
    decision_id: str
    level: int
    target_component: str | None
    transformation: str
    diagnosis: str
    hypothesis: str
    expected_changes: dict[str, str]
    evidence_refs: list[str]
    expert_assessment: dict[str, str] = field(default_factory=dict)
    role_analysis: dict[str, str] = field(default_factory=dict)
    mathematical_plan: dict[str, str] = field(default_factory=dict)


@dataclass
class EpisodeRecord:
    scientific_iteration: int
    candidate_id: str
    score: float
    decision: InterventionDecision | None
    hypothesis_status: str
    evidence_path: str | None = None


@dataclass
class AgentState:
    run_id: str
    phase: Phase = Phase.INIT
    scientific_iteration: int = 0
    engineering_attempts: int = 0
    stagnation_count: int = 0
    current_candidate: RewardCandidate | None = None
    best_candidate: RewardCandidate | None = None
    best_evidence: EvaluationEvidence | None = None
    latest_evidence: EvaluationEvidence | None = None
    pending_decision: InterventionDecision | None = None
    stop_reason: StopReason | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value
