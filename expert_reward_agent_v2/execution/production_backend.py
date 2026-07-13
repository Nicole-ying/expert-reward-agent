from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

from llm_clients.deepseek_client import DeepSeekClient

from ..agent.schemas import (
    ComponentEvidence,
    EnvironmentContract,
    EvaluationEvidence,
    InterventionDecision,
    RewardCandidate,
)
from ..knowledge.ontology import COMPONENT_ROLES, MATHEMATICAL_FORMS


class ProductionBackend:
    def __init__(self, config_path: Path, run_dir: Path, seed: int | None = None):
        self.config_path = config_path.resolve()
        self.repo_root = Path(__file__).resolve().parents[2]
        self.run_dir = run_dir.resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if seed is not None:
            self.config["training"]["seed"] = int(seed)
        llm = self.config["llm"]
        self.client = DeepSeekClient(api_key_env=llm["api_key_env"], base_url=llm["base_url"])
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "generator_system.md"
        core_path = Path(__file__).resolve().parents[1] / "prompts" / "expert_reasoning_core.md"
        self.generator_system = prompt_path.read_text(encoding="utf-8") + "\n\n" + core_path.read_text(encoding="utf-8")
        self.generated = 0
        self.masked_step_source = ""
        self.evaluation_counts: dict[str, int] = {}

    def perceive(self) -> dict[str, Any]:
        inputs = self.config["inputs"]
        task_path = self._path(inputs["task_spec_path"])
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        masked_path = self._path(inputs["masked_step_path"])
        self.masked_step_source = masked_path.read_text(encoding="utf-8")
        perception = {
            "task": task,
            "masked_step_source": self.masked_step_source,
            "source_paths": {"task": str(task_path), "masked_step": str(masked_path)},
        }
        self._write_json(self.run_dir / "inputs" / "perception.json", perception)
        return perception

    def build_contract(self, perception: dict[str, Any]) -> EnvironmentContract:
        task = perception["task"]
        obs = task.get("observation_space", {})
        action = task.get("action_space", {})
        agent = self.config["agent"]
        contract = EnvironmentContract(
            environment_id=str(task.get("env_id", "anonymous_env")),
            task_description=str(task.get("task_description", "")),
            observation_fields=list(obs.get("fields", [])),
            action_description=action,
            termination_conditions=[str(item) for item in task.get("termination_conditions", [])],
            target_score=float(agent["target_score"]),
            max_episode_steps=int(agent.get("max_episode_steps", 1000)),
            available_signals=[str(item.get("name", item.get("index"))) for item in obs.get("fields", [])],
            unknowns=["Exact semantic outcome labels are unavailable unless a verified adapter supplies them."],
        )
        self._write_json(self.run_dir / "inputs" / "environment_contract.json", asdict(contract))
        return contract

    def generate_candidate(
        self,
        contract: EnvironmentContract,
        knowledge: list[dict[str, Any]],
        parent: RewardCandidate | None,
        decision: InterventionDecision | None,
    ) -> RewardCandidate:
        self.generated += 1
        prompt = self._generation_prompt(contract, knowledge, parent, decision)
        errors: list[str] = []
        for attempt in range(int(self.config["agent"].get("generation_retries", 2)) + 1):
            repair = "" if not errors else f"\nPrevious draft validation errors: {errors}. Repair compliance only."
            try:
                raw = self.client.chat(
                    model=self.config["llm"]["model"],
                    system_prompt=self.generator_system,
                    user_prompt=prompt + repair,
                    temperature=float(self.config["llm"].get("generation_temperature", 0.15)),
                    max_tokens=int(self.config["llm"].get("generation_max_tokens", 8192)),
                    json_mode=True,
                )
                data = json.loads(raw)
            except Exception as exc:
                errors = [f"API or JSON error: {type(exc).__name__}: {exc}"]
                continue
            candidate = RewardCandidate(
                candidate_id=f"candidate_{self.generated:02d}",
                code=_extract_code(str(data.get("code", ""))),
                component_roles={str(k): str(v) for k, v in data.get("component_roles", {}).items()},
                mathematical_forms={str(k): str(v) for k, v in data.get("mathematical_forms", {}).items()},
                parent_id=parent.candidate_id if parent else None,
                intervention_id=decision.decision_id if decision else None,
            )
            validation = self.validate_candidate(candidate)
            self._write_generation(candidate, prompt + repair, raw, data, validation, attempt)
            if validation["valid"]:
                return candidate
            errors = validation["errors"]
        raise RuntimeError(f"LLM failed reward validation: {'; '.join(errors)}")

    def validate_candidate(self, candidate: RewardCandidate) -> dict[str, Any]:
        errors: list[str] = []
        signature = "def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):"
        if signature not in candidate.code:
            errors.append("missing exact compute_reward signature")
        body = candidate.code.replace(signature, "")
        forbidden = ["original_reward", "import ", "class ", "try:", "except ", "eval(", "exec(", "open(", "self."]
        errors.extend(f"forbidden token: {token}" for token in forbidden if token in body)
        try:
            tree = ast.parse(candidate.code)
            compile(tree, f"<{candidate.candidate_id}>", "exec")
            functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if len(functions) != 1 or functions[0].name != "compute_reward":
                errors.append("code must contain exactly one compute_reward function")
            if not any(isinstance(node, ast.Return) and isinstance(node.value, (ast.Tuple, ast.List)) for node in ast.walk(tree)):
                errors.append("compute_reward must return reward and components")
        except (SyntaxError, ValueError) as exc:
            errors.append(f"invalid Python: {exc}")
        if not candidate.component_roles:
            errors.append("component_roles is empty")
        if set(candidate.component_roles) != set(candidate.mathematical_forms):
            errors.append("component role and mathematical form keys differ")
        unknown_roles = sorted(set(candidate.component_roles.values()) - COMPONENT_ROLES)
        if unknown_roles:
            errors.append(f"unknown component roles: {unknown_roles}")
        unknown_forms = sorted(set(candidate.mathematical_forms.values()) - MATHEMATICAL_FORMS)
        if unknown_forms:
            errors.append(f"unknown mathematical forms: {unknown_forms}")
        return {"valid": not errors, "errors": errors}

    def train(self, candidate: RewardCandidate, scientific_iteration: int) -> dict[str, Any]:
        iteration_dir = self.run_dir / "artifacts" / f"iter_{scientific_iteration:02d}"
        training_dir = iteration_dir / "training"
        reward_path = iteration_dir / "reward.py"
        reward_path.parent.mkdir(parents=True, exist_ok=True)
        reward_path.write_text(candidate.code, encoding="utf-8")
        training = self.config["training"]
        command = [
            sys.executable, "-m", "training.train_sb3_wrapper",
            "--config", str(self._path(training["legacy_config_path"])),
            "--reward", str(reward_path),
            "--run-name", f"{self.run_dir.name}_iter_{scientific_iteration:02d}",
            "--total-timesteps", str(int(training["total_timesteps"])),
            "--eval-episodes", str(int(training["eval_episodes"])),
            "--eval-seed-offset", str(int(training.get("eval_seed_offset", 10000))),
            "--seed", str(int(training.get("seed", 0))),
            "--save-dir", str(training_dir),
        ]
        completed = subprocess.run(command, cwd=self.repo_root, text=True, capture_output=True, encoding="utf-8", errors="replace")
        (iteration_dir / "training.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (iteration_dir / "training.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"SB3 training failed with exit code {completed.returncode}; see {iteration_dir}")
        return {"training_dir": str(training_dir), "iteration": scientific_iteration}

    def evaluate(self, candidate: RewardCandidate, training_artifact: dict[str, Any]) -> EvaluationEvidence:
        training_dir = Path(training_artifact["training_dir"])
        count = self.evaluation_counts.get(candidate.candidate_id, 0)
        self.evaluation_counts[candidate.candidate_id] = count + 1
        if count == 0:
            summary = json.loads((training_dir / "training_summary.json").read_text(encoding="utf-8"))
            evaluation = json.loads((training_dir / "eval_result.json").read_text(encoding="utf-8"))
        else:
            summary, evaluation = self._confirm_evaluation(training_dir)
        diagnostics = summary.get("evaluation_diagnostics", {})
        return EvaluationEvidence(
            candidate_id=candidate.candidate_id,
            mean_score=float(evaluation.get("mean_eval_reward", 0.0)),
            score_std=float(evaluation.get("std_eval_reward", 0.0)),
            mean_length=float(evaluation.get("mean_episode_length", 0.0)),
            episodes=int(evaluation.get("eval_episodes", 0)),
            termination_counts=_termination_counts(evaluation),
            behavior_metrics=_behavior_metrics(evaluation, diagnostics),
            components=_component_evidence(summary, diagnostics, candidate),
            exploit_flags=_exploit_flags(evaluation, diagnostics),
            source="confirmation_evaluation" if count else "final_policy_evaluation",
        )

    def _confirm_evaluation(self, training_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        training = self.config["training"]
        output_path = training_dir / "confirmation_eval.json"
        command = [
            sys.executable, "-m", "expert_reward_agent_v2.execution.evaluate_saved_model",
            "--config", str(self._path(training["legacy_config_path"])),
            "--model", str(training_dir / "model.zip"),
            "--reward", str(training_dir.parent / "reward.py"),
            "--episodes", str(int(training.get("confirmation_eval_episodes", training["eval_episodes"]))),
            "--seed", str(int(training.get("seed", 0))),
            "--seed-offset", str(int(training.get("confirmation_seed_offset", 20000))),
            "--output", str(output_path),
        ]
        completed = subprocess.run(command, cwd=self.repo_root, text=True, capture_output=True, encoding="utf-8", errors="replace")
        (training_dir / "confirmation.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (training_dir / "confirmation.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"Confirmation evaluation failed; see {training_dir}")
        evaluation = json.loads(output_path.read_text(encoding="utf-8"))
        summary = json.loads((training_dir / "training_summary.json").read_text(encoding="utf-8"))
        return summary, evaluation

    def _generation_prompt(self, contract: EnvironmentContract, knowledge: list[dict[str, Any]], parent: RewardCandidate | None, decision: InterventionDecision | None) -> str:
        return "\n\n".join([
            "Environment contract:\n" + json.dumps(asdict(contract), ensure_ascii=False, indent=2),
            "Masked step source (environment facts only; do not infer hidden benchmark identity):\n" + self.masked_step_source,
            "Expert cards:\n" + json.dumps(knowledge, ensure_ascii=False, indent=2),
            "Intervention plan:\n" + json.dumps(asdict(decision) if decision else {}, ensure_ascii=False, indent=2),
            "Parent candidate:\n" + json.dumps(asdict(parent) if parent else None, ensure_ascii=False, indent=2),
        ])

    def _write_generation(self, candidate: RewardCandidate, prompt: str, raw: str, response: dict[str, Any], validation: dict[str, Any], attempt: int) -> None:
        root = self.run_dir / "artifacts" / "generations" / candidate.candidate_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "system_prompt.md").write_text(self.generator_system, encoding="utf-8")
        (root / f"user_prompt_attempt_{attempt}.md").write_text(prompt, encoding="utf-8")
        (root / f"raw_response_attempt_{attempt}.txt").write_text(raw, encoding="utf-8")
        (root / f"response_attempt_{attempt}.json").write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
        (root / f"validation_attempt_{attempt}.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
        (root / "reward.py").write_text(candidate.code, encoding="utf-8")
        self._write_json(root / "candidate.json", asdict(candidate))

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.repo_root / path


def _extract_code(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, re.S)
    return (match.group(1) if match else text).strip()


def _termination_counts(evaluation: dict[str, Any]) -> dict[str, int]:
    breakdown = evaluation.get("termination_breakdown", {})
    if breakdown:
        return {str(k): int(v) for k, v in breakdown.items()}
    return {
        "terminated": int(evaluation.get("terminated_count", 0)),
        "truncated": int(evaluation.get("truncated_count", 0)),
    }


def _behavior_metrics(evaluation: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "score_min": evaluation.get("min_eval_reward"),
        "score_max": evaluation.get("max_eval_reward"),
        "score_median": evaluation.get("median_eval_reward"),
    }
    metrics.update(diagnostics.get("behavior_metrics", {}))
    return metrics


def _component_evidence(summary: dict[str, Any], diagnostics: dict[str, Any], candidate: RewardCandidate) -> list[ComponentEvidence]:
    final_stats = diagnostics.get("step_component_stats", {})
    episode_stats = diagnostics.get("episode_component_sum_stats", {})
    if not final_stats:
        final_eval = summary.get("external_eval", {}).get("final_policy_component_evaluation", {})
        final_stats = final_eval.get("component_stats", {})
        episode_stats = final_eval.get("episode_component_sum_stats", {})
    if not final_stats:
        final_stats = summary.get("component_summary", {}).get("component_stats", {})
        episode_stats = summary.get("component_summary", {}).get("episode_component_sum_stats", {})
    output: list[ComponentEvidence] = []
    for name in candidate.component_roles:
        stats = final_stats.get(name, final_stats.get(f"component.{name}", {}))
        episode = episode_stats.get(name, {})
        output.append(ComponentEvidence(
            name=name,
            role=candidate.component_roles[name],
            mean=float(stats.get("mean", 0.0)),
            abs_mean=float(stats.get("abs_mean", 0.0)),
            episode_sum_mean=float(episode.get("mean", 0.0)),
            active_rate=float(stats.get("nonzero_rate", stats.get("active_rate", 0.0))),
            mean_when_active=float(stats.get("mean_when_active", 0.0)),
        ))
    return output


def _exploit_flags(evaluation: dict[str, Any], diagnostics: dict[str, Any]) -> list[str]:
    flags = diagnostics.get("exploit_flags", [])
    return [str(item) for item in flags]
