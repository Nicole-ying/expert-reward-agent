from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from expert_reward_agent_v2.agent.controller import RewardDesignAgent
from expert_reward_agent_v2.agent.planner import RuleBasedExpertPlanner
from expert_reward_agent_v2.agent.schemas import Phase, StopReason
from expert_reward_agent_v2.agent.state_machine import InvalidTransition, transition
from expert_reward_agent_v2.behavior import GeneratedBehaviorAdapter, GenericBehaviorAdapter
from expert_reward_agent_v2.execution.mock_backend import MockBackend
from expert_reward_agent_v2.memory import EpisodicMemory, WorkingMemory


ROOT = Path(__file__).resolve().parents[1]


class StateMachineTests(unittest.TestCase):
    def test_valid_and_invalid_transition(self) -> None:
        self.assertEqual(transition(Phase.INIT, Phase.PERCEIVE), Phase.PERCEIVE)
        with self.assertRaises(InvalidTransition):
            transition(Phase.INIT, Phase.TRAIN)


class MemoryTests(unittest.TestCase):
    def test_memory_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            working = WorkingMemory(root / "working.json")
            working.save({"iteration": 2})
            self.assertEqual(working.load()["iteration"], 2)
            episodic = EpisodicMemory(root / "episodes.jsonl")
            episodic.append({"score": 10})
            episodic.append({"score": 20})
            self.assertEqual([item["score"] for item in episodic.all()], [10, 20])


class BehaviorTests(unittest.TestCase):
    def test_generic_and_generated_metrics(self) -> None:
        trajectories = [{
            "observations": [[3.0], [2.0], [1.0]],
            "actions": [1, 1],
            "rewards": [1.0, 2.0],
            "terminated": True,
        }]
        self.assertEqual(GenericBehaviorAdapter().extract_metrics(trajectories)["return_mean"], 3.0)
        adapter = GeneratedBehaviorAdapter({"metrics": [{
            "name": "distance_change",
            "operation": "delta",
            "field": "observations",
            "index": 0,
            "evidence_kind": "proxy",
            "confidence": 0.8,
        }]})
        metrics = adapter.extract_metrics(trajectories)
        self.assertEqual(metrics["distance_change"], -2.0)
        self.assertEqual(metrics["distance_change__evidence"], "proxy")


class EndToEndTests(unittest.TestCase):
    def test_mock_agent_solves_and_audits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            agent = RewardDesignAgent(
                backend=MockBackend(),
                planner=RuleBasedExpertPlanner(),
                run_dir=run_dir,
                knowledge_dir=ROOT / "knowledge" / "cards",
            )
            state = agent.run()
            self.assertEqual(state.stop_reason, StopReason.SOLVED_CONFIRMED)
            self.assertEqual(state.scientific_iteration, 3)
            self.assertEqual(state.best_evidence.mean_score, 225.0)
            calls = [json.loads(line) for line in (run_dir / "audit" / "tool_calls.jsonl").read_text().splitlines()]
            self.assertTrue(any(item["tool_name"] == "train" for item in calls))
            self.assertTrue(any(item["tool_name"] == "diagnose" for item in calls))
            self.assertEqual(len(agent.episodic.all()), 3)
            levels = [item["decision"]["level"] for item in agent.episodic.all()]
            self.assertEqual(levels, [0, 1, 2])
            for item in agent.episodic.all():
                assessment = item["decision"]["expert_assessment"]
                self.assertEqual(
                    set(assessment),
                    {"reachability", "directionality", "scale", "credit_assignment", "task_alignment", "exploit_resistance"},
                )
                self.assertTrue(item["decision"]["role_analysis"])
                self.assertTrue(item["decision"]["mathematical_plan"])


if __name__ == "__main__":
    unittest.main()
