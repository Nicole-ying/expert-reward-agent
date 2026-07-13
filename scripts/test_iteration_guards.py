import tempfile
import unittest
from pathlib import Path

from pipeline.run_iterative_experiment import find_identical_historical_reward


REWARD_A = """def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    reward = float(obs[0] - next_obs[0])
    return reward, {"progress": reward}
"""

REWARD_B = """def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    reward = -float(next_obs[0] ** 2)
    return reward, {"distance": reward}
"""

REWARD_C = """def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    reward = -abs(float(next_obs[0]))
    return reward, {"distance": reward}
"""


class HistoricalDuplicateTests(unittest.TestCase):
    def test_matches_any_earlier_iteration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = {"experiment": {"run_root": str(root)}}
            seed_root = root / "paper" / "seed_0"
            for iteration, code in [(1, REWARD_A), (2, REWARD_B)]:
                path = seed_root / f"iter_{iteration:02d}" / "generation" / f"reward_v{iteration}.py"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(code, encoding="utf-8")

            current = seed_root / "iter_03" / "generation" / "reward_v3.py"
            current.parent.mkdir(parents=True, exist_ok=True)
            current.write_text(REWARD_A, encoding="utf-8")
            match = find_identical_historical_reward(cfg, "paper", 0, 3, current)
            self.assertEqual(match[0], 1)

            current.write_text(REWARD_B, encoding="utf-8")
            match = find_identical_historical_reward(cfg, "paper", 0, 3, current)
            self.assertEqual(match[0], 2)

            current.write_text(REWARD_C, encoding="utf-8")
            self.assertIsNone(find_identical_historical_reward(cfg, "paper", 0, 3, current))


if __name__ == "__main__":
    unittest.main()
