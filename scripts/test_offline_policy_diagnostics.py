import unittest

from scripts.offline_policy_diagnostics import ScalarStats, infer_additive_terms, parse_reward_output


class OfflineDiagnosticsTests(unittest.TestCase):
    def test_active_statistics_do_not_hide_signed_cancellation(self):
        stats = ScalarStats()
        for value in (1.0, -1.0, 0.0):
            stats.add(value)
        summary = stats.summary()
        self.assertAlmostEqual(summary["mean"], 0.0)
        self.assertAlmostEqual(summary["abs_mean"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["nonzero_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["abs_mean_when_active"], 1.0)

    def test_legacy_additive_inference_separates_modulator(self):
        records = [
            {"generated_reward": 3.0, "progress": 1.0, "bonus": 2.5, "penalty": -0.5, "gate": 0.8},
            {"generated_reward": 2.0, "progress": 0.5, "bonus": 2.0, "penalty": -0.5, "gate": 0.4},
        ]
        result = infer_additive_terms(records)
        self.assertEqual(result["status"], "exact")
        self.assertEqual(set(result["reward_terms"]), {"progress", "bonus", "penalty"})
        self.assertEqual(result["diagnostics"], ["gate"])

    def test_nested_future_schema_is_backward_compatible(self):
        generated, terms = parse_reward_output(
            (1.5, {"reward_terms": {"progress": 1.5}, "diagnostics": {"gate": 0.2}})
        )
        self.assertEqual(generated, 1.5)
        self.assertEqual(terms, {"progress": 1.5, "diagnostic.gate": 0.2})


if __name__ == "__main__":
    unittest.main()
