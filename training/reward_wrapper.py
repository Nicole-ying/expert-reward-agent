import importlib.util
from pathlib import Path
import gymnasium as gym


def load_reward_function(path: str):
    spec = importlib.util.spec_from_file_location("generated_reward_module", str(Path(path)))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.compute_reward


class RewardOverrideWrapper(gym.Wrapper):
    def __init__(self, env, reward_fn, max_training_steps_for_progress=100000):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.max_training_steps_for_progress = max(1, int(max_training_steps_for_progress))
        self.global_step = 0
        self.last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info

    def step(self, action):
        obs = self.last_obs
        next_obs, original_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        self.global_step += 1
        progress = min(1.0, self.global_step / self.max_training_steps_for_progress)

        safe_info = dict(info)
        safe_info["terminated"] = bool(terminated)
        safe_info["truncated"] = bool(truncated)
        safe_info["done"] = bool(terminated or truncated)

        reward_out = self.reward_fn(obs, action, next_obs, original_reward, safe_info, training_progress=progress)

        reward_terms = None
        if isinstance(reward_out, tuple) and len(reward_out) == 2:
            generated_reward, reward_terms = reward_out
        elif isinstance(reward_out, list) and len(reward_out) == 2:
            generated_reward, reward_terms = reward_out
        else:
            generated_reward = reward_out

        if isinstance(reward_terms, dict):
            safe_info["reward_terms"] = dict(reward_terms)
        elif "reward_terms" not in safe_info:
            safe_info["reward_terms"] = {}

        safe_info["original_env_reward"] = float(original_reward)
        safe_info["generated_reward"] = float(generated_reward)
        safe_info["training_progress"] = float(progress)
        safe_info["reward_terms"]["total_reward"] = float(generated_reward)

        self.last_obs = next_obs
        return next_obs, float(generated_reward), terminated, truncated, safe_info
