import unittest

import numpy as np

from scripts.offline_policy_diagnostics import BipedalDiagnostics, LanderDiagnostics
from training.train_sb3_wrapper import RewardComponentStatsCallback


class DiagnosticPipelineTests(unittest.TestCase):
    def test_component_callback_reports_active_and_episode_statistics(self):
        callback = RewardComponentStatsCallback()
        callback._update_value("component.progress", 1.0)
        callback._update_value("component.progress", 0.0)
        callback.episode_component_sums = [{"progress": 2.0}, {"progress": -1.0}]
        summary = callback.summary()
        stats = summary["component_stats"]["component.progress"]
        self.assertEqual(stats["nonzero_rate"], 0.5)
        self.assertEqual(stats["mean_when_active"], 1.0)
        self.assertEqual(summary["episode_component_sum_stats"]["progress"]["mean"], 0.5)

    def test_lander_adapter_reports_facts_without_claiming_crash(self):
        adapter = LanderDiagnostics()
        adapter.observe(np.array([0.1, 0.2, 0.0, -0.1, 0.02, 0.0, 1.0, 1.0]), 2)
        result = adapter.summary(terminated=True, truncated=False)
        self.assertEqual(result["outcome"], "terminated_unclassified")
        self.assertEqual(result["termination_reason"], "unknown")
        self.assertEqual(result["action_rate_2"], 1.0)

    def test_bipedal_adapter_uses_declared_indices(self):
        adapter = BipedalDiagnostics()
        obs = np.zeros(24)
        obs[0] = 0.1
        obs[2] = 0.3
        obs[12] = 1.0
        adapter.observe(obs, np.array([0.1, -0.2, 0.3, -0.4]))
        result = adapter.summary(terminated=False, truncated=True)
        self.assertEqual(result["outcome"], "timeout")
        self.assertAlmostEqual(result["mean_horizontal_velocity"], 0.3)
        self.assertAlmostEqual(result["left_contact_rate"], 1.0)
        self.assertEqual(result["termination_reason"], "unknown")


if __name__ == "__main__":
    unittest.main()
