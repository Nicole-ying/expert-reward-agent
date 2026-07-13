from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from ..execution.interfaces import ExecutionBackend, ExpertPlanner
from ..memory import CaseMemory, EpisodicMemory, SemanticMemory, WorkingMemory
from ..tools import ToolRegistry
from .schemas import AgentState, EnvironmentContract, EvaluationEvidence, Phase, StopReason
from .state_machine import transition
from .stop_policy import StopPolicy


class RewardDesignAgent:
    def __init__(
        self,
        backend: ExecutionBackend,
        planner: ExpertPlanner,
        run_dir: Path,
        knowledge_dir: Path,
        stop_policy: StopPolicy | None = None,
        max_engineering_attempts: int = 3,
    ):
        self.backend = backend
        self.planner = planner
        self.run_dir = run_dir
        self.stop_policy = stop_policy or StopPolicy()
        self.max_engineering_attempts = max_engineering_attempts
        self.state = AgentState(run_id=run_dir.name)
        self.working = WorkingMemory(run_dir / "memory" / "working.json")
        self.episodic = EpisodicMemory(run_dir / "memory" / "episodes.jsonl")
        self.semantic = SemanticMemory(knowledge_dir)
        self.cases = CaseMemory(run_dir.parent / "case_memory.jsonl")
        self.tools = ToolRegistry(run_dir / "audit" / "tool_calls.jsonl")
        self._register_tools()

    def _register_tools(self) -> None:
        self.tools.register("perceive", self.backend.perceive, {Phase.PERCEIVE})
        self.tools.register("build_contract", self.backend.build_contract, {Phase.CONTRACT})
        self.tools.register("plan_initial", self.planner.plan_initial, {Phase.PLAN})
        self.tools.register("generate", self.backend.generate_candidate, {Phase.GENERATE})
        self.tools.register("validate", self.backend.validate_candidate, {Phase.VALIDATE})
        self.tools.register("train", self.backend.train, {Phase.TRAIN})
        self.tools.register("evaluate", self.backend.evaluate, {Phase.EVALUATE, Phase.CONFIRM})
        self.tools.register("diagnose", self.planner.diagnose, {Phase.DIAGNOSE})

    def run(self) -> AgentState:
        self._move(Phase.PERCEIVE)
        perception = self._require("perceive", self.state.phase)
        self._move(Phase.CONTRACT)
        contract: EnvironmentContract = self._require("build_contract", self.state.phase, perception=perception)
        terms = [contract.task_description, *contract.available_signals]
        knowledge = self.semantic.retrieve(terms)
        inputs_dir = self.run_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        (inputs_dir / "retrieved_knowledge.json").write_text(
            json.dumps(knowledge, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._move(Phase.PLAN)
        self.state.pending_decision = self._require(
            "plan_initial", self.state.phase, contract=contract, knowledge=knowledge
        )

        while True:
            self._move(Phase.GENERATE)
            candidate = self._require(
                "generate",
                self.state.phase,
                contract=contract,
                knowledge=knowledge,
                parent=self.state.current_candidate,
                decision=self.state.pending_decision,
            )
            self.state.current_candidate = candidate
            self._move(Phase.VALIDATE)
            if not self._validate_with_retry(candidate):
                return self._fail(StopReason.ENGINEERING_FAILURE)

            self._move(Phase.TRAIN)
            artifact = self._require("train", self.state.phase, candidate=candidate, scientific_iteration=self.state.scientific_iteration + 1)
            self._move(Phase.EVALUATE)
            evidence: EvaluationEvidence = self._require("evaluate", self.state.phase, candidate=candidate, training_artifact=artifact)
            self.state.scientific_iteration += 1
            self.state.latest_evidence = evidence
            self._update_best(candidate, evidence)
            self._record_episode(evidence)

            if evidence.mean_score >= contract.target_score and not evidence.exploit_flags:
                self._move(Phase.CONFIRM)
                confirmed = self._require("evaluate", self.state.phase, candidate=candidate, training_artifact=artifact)
                reason = self.stop_policy.after_evaluation(self.state, contract, confirmed, confirmed=True)
                if reason:
                    self.state.stop_reason = reason
                    self._move(Phase.STOPPED)
                    self._save()
                    if reason == StopReason.SOLVED_CONFIRMED:
                        self.cases.append({"contract": asdict(contract), "candidate": asdict(candidate), "evidence": asdict(confirmed)})
                    return self.state
                self._move(Phase.DIAGNOSE)
            else:
                reason = self.stop_policy.after_evaluation(self.state, contract, evidence)
                if reason:
                    self.state.stop_reason = reason
                    self._move(Phase.STOPPED)
                    self._save()
                    return self.state
                self._move(Phase.DIAGNOSE)

            self.state.pending_decision = self._require(
                "diagnose",
                self.state.phase,
                contract=contract,
                current=candidate,
                evidence=evidence,
                history=self.episodic.all(),
            )
            self._move(Phase.INTERVENE)
            self._save()

    def _validate_with_retry(self, candidate: Any) -> bool:
        for _ in range(self.max_engineering_attempts):
            result = self.tools.call("validate", self.state.phase, candidate=candidate)
            if result.ok and result.output.get("valid"):
                return True
            self.state.engineering_attempts += 1
        return False

    def _update_best(self, candidate: Any, evidence: EvaluationEvidence) -> None:
        previous = self.state.best_evidence.mean_score if self.state.best_evidence else float("-inf")
        if evidence.mean_score > previous:
            self.state.best_candidate = candidate
            self.state.best_evidence = evidence
            self.state.stagnation_count = 0
        else:
            self.state.stagnation_count += 1

    def _record_episode(self, evidence: EvaluationEvidence) -> None:
        self.episodic.append({
            "scientific_iteration": self.state.scientific_iteration,
            "candidate_id": evidence.candidate_id,
            "score": evidence.mean_score,
            "decision": asdict(self.state.pending_decision) if self.state.pending_decision else None,
            "hypothesis_status": "observed",
        })
        self._save()

    def _require(self, name: str, phase: Phase, **kwargs: Any) -> Any:
        attempts = self.max_engineering_attempts if name in {"perceive", "plan_initial", "generate", "evaluate", "diagnose"} else 1
        last_error = None
        for _ in range(attempts):
            result = self.tools.call(name, phase, **kwargs)
            if result.ok:
                return result.output
            self.state.engineering_attempts += 1
            last_error = result.error
            self._save()
        raise RuntimeError(f"Tool {name} failed: {last_error}")

    def _move(self, target: Phase) -> None:
        self.state.phase = transition(self.state.phase, target)
        self._save()

    def _save(self) -> None:
        payload = self.state.to_dict()
        self.working.save(payload)
        WorkingMemory(self.run_dir / "agent_state.json").save(payload)

    def _fail(self, reason: StopReason) -> AgentState:
        self.state.stop_reason = reason
        self._move(Phase.FAILED)
        return self.state
