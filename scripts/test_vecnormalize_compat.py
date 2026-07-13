from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from training.train_sb3_wrapper import evaluate_model_on_original_env


class RecordingPolicy:
    def __init__(self):
        self.observations = []

    def predict(self, obs, deterministic=True):
        self.observations.append(np.asarray(obs).copy())
        return 0, None


def test_existing_configs_explicitly_disable_normalization():
    for config_path in ["configs/env001_deepseek_rag.yaml", "configs/env002_deepseek_rag.yaml"]:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        assert cfg["training"]["normalize_obs"] is False
        assert cfg["training"]["normalize_reward"] is False


def test_normalized_policy_input_keeps_external_reward_raw(tmp_path):
    vec_env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    normalizer = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    normalizer.obs_rms.mean = np.array([10.0, 10.0, 10.0, 10.0])
    normalizer.obs_rms.var = np.ones(4)
    normalizer.training = False

    stats_path = tmp_path / "vecnormalize.pkl"
    normalizer.save(str(stats_path))
    loaded_env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    loaded = VecNormalize.load(str(stats_path), loaded_env)
    assert loaded.training is False
    assert loaded.norm_reward is False
    loaded.training = False
    loaded.norm_reward = False

    policy = RecordingPolicy()
    result = evaluate_model_on_original_env(
        policy,
        "CartPole-v1",
        eval_episodes=1,
        seed=0,
        eval_seed_offset=10000,
        observation_normalizer=loaded,
    )
    assert policy.observations
    assert np.all(policy.observations[0] < -9.0)
    assert result["episode_rewards"][0] == result["episode_lengths"][0]
    normalizer.close()
    loaded.close()
