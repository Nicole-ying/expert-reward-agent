from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_clients.deepseek_client import DeepSeekClient

from ..agent.schemas import EnvironmentContract, EvaluationEvidence, InterventionDecision, RewardCandidate
from ..knowledge.ontology import COMPONENT_ROLES, EXPERT_DIMENSIONS, MATHEMATICAL_FORMS, STRUCTURAL_TRANSFORMATIONS


class DeepSeekExpertPlanner:
    def __init__(self, model: str, api_key_env: str, base_url: str, temperature: float = 0.1, max_tokens: int = 4096, record_dir: Path | None = None):
        self.client = DeepSeekClient(api_key_env=api_key_env, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "planner_system.md"
        core_path = Path(__file__).resolve().parents[1] / "prompts" / "expert_reasoning_core.md"
        self.system_prompt = prompt_path.read_text(encoding="utf-8") + "\n\n" + core_path.read_text(encoding="utf-8")
        self.record_dir = record_dir
        self.call_index = 0

    def plan_initial(self, contract: EnvironmentContract, knowledge: list[dict[str, Any]]) -> InterventionDecision:
        prompt = f"""Plan the first reward design hypothesis.
Environment contract:
{json.dumps(asdict(contract), ensure_ascii=False, indent=2)}
Retrieved expert cards (open design primitives, not a closed menu):
{json.dumps(knowledge, ensure_ascii=False, indent=2)}
Return keys: diagnosis, hypothesis, transformation, target_component, expected_changes, evidence_refs,
expert_assessment, role_analysis, and mathematical_plan.
expert_assessment must contain reachability, directionality, scale, credit_assignment, task_alignment,
and exploit_resistance. Each value must state the evidence or explicitly say unknown.
role_analysis maps the smallest required functional roles to present, missing, deferred, or unknown with a reason.
mathematical_plan maps each proposed component name to one primary mathematical form from the expert reasoning core.
Choose forms only after deciding component roles and explain the role-to-form reason in the value when needed
using `form: rationale`.
Use transformation=compose_initial_reward. target_component may be null.
"""
        return self._decision_with_retry(prompt, "initial_plan", fixed_level=0)

    def diagnose(
        self,
        contract: EnvironmentContract,
        current: RewardCandidate,
        evidence: EvaluationEvidence,
        history: list[dict[str, Any]],
    ) -> InterventionDecision:
        prompt = f"""Diagnose one trained reward candidate and choose one intervention.
Answer these questions inside diagnosis: what behavior occurred, which component or missing role is
the best intervention target, and what happened after the previous intervention.

Environment contract:
{json.dumps(asdict(contract), ensure_ascii=False, indent=2)}
Current reward metadata and code:
{json.dumps(asdict(current), ensure_ascii=False, indent=2)}
Final-policy evaluation evidence (primary evidence):
{json.dumps(asdict(evidence), ensure_ascii=False, indent=2)}
Episodic history:
{json.dumps(history[-6:], ensure_ascii=False, indent=2)}

First test signal-role completeness. Choose Level 1 only for an isolated sign/scale defect; choose
Level 2 for missing/unreachable roles, weak credit assignment, wrong temporal semantics, or a failed
Level-1 repair; choose Level 3 only after documented structural stagnation.
Return keys: level (1, 2, or 3), diagnosis, hypothesis, transformation, target_component,
expected_changes, evidence_refs, expert_assessment, role_analysis, and mathematical_plan.
expert_assessment must separately answer reachability, directionality, scale, credit_assignment,
task_alignment, and exploit_resistance using evidence or unknown. role_analysis maps relevant functional
roles to present, missing, deferred, or unknown with reasons. expected_changes must contain falsifiable predictions.
mathematical_plan maps each retained or changed component to `primary_form: rationale`.
"""
        return self._decision_with_retry(prompt, "diagnosis")

    def _decision_with_retry(self, prompt: str, call_kind: str, fixed_level: int | None = None) -> InterventionDecision:
        repair = ""
        last_error: Exception | None = None
        for _ in range(3):
            data = self._json(prompt + repair, call_kind)
            try:
                level = fixed_level if fixed_level is not None else int(data.get("level", 1))
                if level not in {0, 1, 2, 3} or (fixed_level is None and level == 0):
                    raise ValueError(f"Invalid intervention level from LLM: {level}")
                return _decision(data, level=level)
            except (TypeError, ValueError) as exc:
                last_error = exc
                repair = (
                    "\n\nYour previous JSON violated the decision schema: " + str(exc) +
                    ". Return a complete corrected JSON object; do not omit unknown dimensions."
                )
        raise RuntimeError(f"Planner response failed decision schema: {last_error}")

    def _json(self, prompt: str, call_kind: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                raw = self.client.chat(
                    model=self.model,
                    system_prompt=self.system_prompt,
                    user_prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    json_mode=True,
                )
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("Planner response must be a JSON object")
                self._record(call_kind, attempt, prompt, raw, data)
                return data
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                self._record(call_kind, attempt, prompt, locals().get("raw", ""), {"parse_error": str(exc)})
        raise RuntimeError(f"Planner did not return valid JSON: {last_error}")

    def _record(self, call_kind: str, attempt: int, user_prompt: str, raw: str, parsed: dict[str, Any]) -> None:
        if self.record_dir is None:
            return
        self.call_index += 1
        root = self.record_dir / f"{self.call_index:03d}_{call_kind}_attempt_{attempt}"
        root.mkdir(parents=True, exist_ok=True)
        (root / "system_prompt.md").write_text(self.system_prompt, encoding="utf-8")
        (root / "user_prompt.md").write_text(user_prompt, encoding="utf-8")
        (root / "raw_response.txt").write_text(raw, encoding="utf-8")
        (root / "parsed_response.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")


def _decision(data: dict[str, Any], level: int) -> InterventionDecision:
    expected = data.get("expected_changes", {})
    if not isinstance(expected, dict):
        raise ValueError("expected_changes must be an object")
    refs = data.get("evidence_refs", [])
    if not isinstance(refs, list):
        raise ValueError("evidence_refs must be a list")
    assessment = data.get("expert_assessment", {})
    if not isinstance(assessment, dict):
        raise ValueError("expert_assessment must be an object")
    missing_dimensions = [name for name in EXPERT_DIMENSIONS if not str(assessment.get(name, "")).strip()]
    if missing_dimensions:
        raise ValueError(f"expert_assessment is missing: {missing_dimensions}")
    role_analysis = data.get("role_analysis", {})
    if not isinstance(role_analysis, dict) or not role_analysis:
        raise ValueError("role_analysis must be a non-empty object")
    unknown_role_keys = sorted(set(role_analysis) - COMPONENT_ROLES)
    if unknown_role_keys:
        raise ValueError(f"role_analysis contains unknown roles: {unknown_role_keys}")
    mathematical_plan = data.get("mathematical_plan", {})
    if not isinstance(mathematical_plan, dict) or not mathematical_plan:
        raise ValueError("mathematical_plan must be a non-empty object")
    invalid_plans = []
    for component, description in mathematical_plan.items():
        primary_form = str(description).split(":", 1)[0].strip()
        if primary_form not in MATHEMATICAL_FORMS:
            invalid_plans.append(f"{component}={primary_form}")
    if invalid_plans:
        raise ValueError(f"mathematical_plan contains unknown forms: {invalid_plans}")
    transformation = str(data.get("transformation", "unspecified"))
    if transformation not in STRUCTURAL_TRANSFORMATIONS:
        raise ValueError(f"Unknown structural transformation: {transformation}")
    return InterventionDecision(
        decision_id=f"decision_{uuid.uuid4().hex[:8]}",
        level=level,
        target_component=data.get("target_component"),
        transformation=transformation,
        diagnosis=str(data.get("diagnosis", "")),
        hypothesis=str(data.get("hypothesis", "")),
        expected_changes={str(key): str(value) for key, value in expected.items()},
        evidence_refs=[str(item) for item in refs],
        expert_assessment={str(key): str(value) for key, value in assessment.items()},
        role_analysis={str(key): str(value) for key, value in role_analysis.items()},
        mathematical_plan={str(key): str(value) for key, value in mathematical_plan.items()},
    )
