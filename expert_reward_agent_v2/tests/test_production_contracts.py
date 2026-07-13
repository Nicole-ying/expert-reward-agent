from __future__ import annotations

import unittest

from expert_reward_agent_v2.agent.schemas import RewardCandidate
from expert_reward_agent_v2.execution.production_backend import ProductionBackend


class ProductionContractTests(unittest.TestCase):
    def test_reward_validator_accepts_contract(self) -> None:
        backend = ProductionBackend.__new__(ProductionBackend)
        candidate = RewardCandidate(
            candidate_id="test",
            code=(
                "def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):\n"
                "    progress = float(obs[0] - next_obs[0])\n"
                "    total_reward = progress\n"
                "    components = {'progress': progress}\n"
                "    return float(total_reward), components\n"
            ),
            component_roles={"progress": "process_progress"},
            mathematical_forms={"progress": "state_improvement"},
        )
        self.assertTrue(backend.validate_candidate(candidate)["valid"])

    def test_reward_validator_rejects_original_reward(self) -> None:
        backend = ProductionBackend.__new__(ProductionBackend)
        candidate = RewardCandidate(
            candidate_id="bad",
            code=(
                "def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):\n"
                "    total_reward = original_reward\n"
                "    components = {'leak': total_reward}\n"
                "    return float(total_reward), components\n"
            ),
            component_roles={"leak": "task_completion"},
            mathematical_forms={"leak": "hidden_reward"},
        )
        result = backend.validate_candidate(candidate)
        self.assertFalse(result["valid"])
        self.assertTrue(any("original_reward" in item for item in result["errors"]))
