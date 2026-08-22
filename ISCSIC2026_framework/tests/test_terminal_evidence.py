from training.train_sb3_wrapper import write_training_feedback_md


def test_episode_terminal_audit_aligns_native_outcome_and_components(tmp_path):
    output = tmp_path / "training_feedback.md"
    eval_result = {
        "eval_episodes": 2,
        "eval_seed_offset": 10000,
        "eval_seeds": [10000, 10001],
        "episode_rewards": [-80.0, 10.0],
        "episode_lengths": [70, 1000],
        "episode_terminated": [True, False],
        "episode_final_observations": [[0.2] * 8, [0.1] * 8],
        "episode_component_sums": [
            {"terminal_event": 10.0, "progress": 1.0},
            {"terminal_event": 0.0, "progress": 1.0},
        ],
        "mean_eval_reward": -35.0,
        "mean_episode_length": 535.0,
        "min_eval_reward": -80.0,
        "max_eval_reward": 10.0,
        "termination_breakdown": {"terminated": 1, "truncated": 1},
        "final_policy_component_error_count": 0,
        "final_policy_component_evaluation": {
            "terminal_event": {
                "episode_sum_mean": 5.0,
                "signed_contribution_share": 0.9,
                "magnitude_share": 0.9,
                "active_rate": 0.01,
            },
            "progress": {
                "episode_sum_mean": 1.0,
                "signed_contribution_share": 0.1,
                "magnitude_share": 0.1,
                "active_rate": 1.0,
            },
        },
    }
    component_summary = {"reward_error_count_max": 0}

    write_training_feedback_md(output, {}, eval_result, component_summary)
    text = output.read_text(encoding="utf-8")

    assert "## Episode-level terminal audit" in text
    assert "positive on non-positive native outcomes=1/2" in text
    assert "possible horizon/proxy accumulation" in text
    assert "`[0.200, 0.200, 0.200" in text
