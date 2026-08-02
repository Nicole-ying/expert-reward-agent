import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import run_iterative_experiment as controller


class CandidateRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_candidate(self, valid, errors):
        cfg = {"experiment": {"run_root": str(self.root)}}
        run_name = "case/seed_0/iter_02/generation"
        generation_dir = self.root / run_name
        validation_dir = generation_dir / "validations"
        validation_dir.mkdir(parents=True)
        (generation_dir / "reward_v2.py").write_text(
            "def compute_reward(obs, action, next_obs, terminated, truncated, info):\n"
            "    return 0.0, {'progress': 0.0}\n",
            encoding="utf-8",
        )
        (generation_dir / "reward_v2.md").write_text(
            "rejected draft\n", encoding="utf-8"
        )
        validation_path = validation_dir / "reward_v2.validation.json"
        validation_path.write_text(
            json.dumps({"valid": valid, "errors": errors}), encoding="utf-8"
        )
        return cfg, run_name, validation_path

    def test_retry_receives_exact_latest_error_and_archives_draft(self):
        cfg, run_name, validation_path = self.make_candidate(
            False, ["nested helper function phi is forbidden"]
        )
        received = []

        def command_factory(error_text, retry_number):
            received.append((error_text, retry_number))
            return ["repair-candidate"]

        def fake_run_cmd(command):
            self.assertEqual(command, ["repair-candidate"])
            validation_path.write_text(
                json.dumps({"valid": True, "errors": []}), encoding="utf-8"
            )

        with patch.object(controller, "run_cmd", fake_run_cmd):
            valid, error = controller.validate_candidate_with_retries(
                cfg, run_name, 2, command_factory, "duplicate-retry candidate"
            )

        self.assertTrue(valid)
        self.assertIsNone(error)
        self.assertEqual(len(received), 1)
        self.assertIn("nested helper function phi is forbidden", received[0][0])
        self.assertEqual(received[0][1], 1)
        archives = list(
            (self.root / run_name / "rejected_attempts").glob("attempt_*")
        )
        self.assertEqual(len(archives), 1)
        record = json.loads(
            (archives[0] / "rejection.json").read_text(encoding="utf-8")
        )
        self.assertEqual(record["rejection_type"], "validation")
        self.assertEqual(record["stage"], "duplicate-retry candidate")
        self.assertTrue((archives[0] / "reward_v2.py").exists())

    def test_retry_uses_new_error_from_each_rejected_repair(self):
        cfg, run_name, validation_path = self.make_candidate(False, ["empty code"])
        received = []

        def command_factory(error_text, retry_number):
            received.append(error_text)
            return [str(retry_number)]

        def fake_run_cmd(command):
            if command == ["1"]:
                validation_path.write_text(
                    json.dumps(
                        {"valid": False, "errors": ["auxiliary function is forbidden"]}
                    ),
                    encoding="utf-8",
                )
            else:
                validation_path.write_text(
                    json.dumps({"valid": True, "errors": []}), encoding="utf-8"
                )

        with patch.object(controller, "run_cmd", fake_run_cmd):
            valid, error = controller.validate_candidate_with_retries(
                cfg,
                run_name,
                2,
                command_factory,
                "fresh duplicate-recovery candidate",
            )

        self.assertTrue(valid)
        self.assertIsNone(error)
        self.assertIn("empty code", received[0])
        self.assertIn("auxiliary function is forbidden", received[1])
        archives = list(
            (self.root / run_name / "rejected_attempts").glob("attempt_*")
        )
        self.assertEqual(len(archives), 2)

    def test_ast_duplicate_detection_ignores_comments_and_formatting(self):
        first = self.root / "first.py"
        second = self.root / "second.py"
        first.write_text(
            "def compute_reward(obs, action, next_obs, terminated, truncated, info):\n"
            "    return 0.0, {'progress': 0.0}\n",
            encoding="utf-8",
        )
        second.write_text(
            "# cosmetic change only\n\n"
            "def compute_reward(obs, action, next_obs, terminated, truncated, info):\n"
            "    return 0.0, {\"progress\": 0.0}\n",
            encoding="utf-8",
        )
        self.assertTrue(controller.is_identical_reward(first, second))


if __name__ == "__main__":
    unittest.main()
