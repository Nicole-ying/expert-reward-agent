import importlib.util
import math
from pathlib import Path
import gymnasium as gym


def load_reward_function(path: str):
    spec = importlib.util.spec_from_file_location("generated_reward_module", str(Path(path)))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.compute_reward


class RewardOverrideWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        reward_fn,
        max_training_steps_for_progress=100000,
        reward_clip=20.0,
        error_fallback="zero",
    ):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.max_training_steps_for_progress = max(1, int(max_training_steps_for_progress))
        self.reward_clip = reward_clip
        self.error_fallback = error_fallback
        self.global_step = 0
        self.last_obs = None
        self.reward_error_count = 0
        self.reward_last_error = ""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info

    def _fallback_reward(self, original_reward):
        if self.error_fallback == "original":
            return float(original_reward)
        return 0.0

    def _parse_reward_output(self, reward_out):
        reward_terms = None
        if isinstance(reward_out, tuple) and len(reward_out) == 2:
            generated_reward, reward_terms = reward_out
        elif isinstance(reward_out, list) and len(reward_out) == 2:
            generated_reward, reward_terms = reward_out
        else:
            generated_reward = reward_out
        return generated_reward, reward_terms

    def _safe_reward(self, obs, action, next_obs, original_reward, safe_info, progress):
        try:
            reward_out = self.reward_fn(obs, action, next_obs, original_reward, safe_info, training_progress=progress)
            generated_reward, reward_terms = self._parse_reward_output(reward_out)
            generated_reward = float(generated_reward)
            if not math.isfinite(generated_reward):
                raise ValueError("non-finite generated reward")
        except Exception as exc:
            self.reward_error_count += 1
            self.reward_last_error = f"{type(exc).__name__}: {exc}"
            generated_reward = self._fallback_reward(original_reward)
            reward_terms = {"reward_error_fallback": generated_reward}

        if self.reward_clip is not None:
            limit = abs(float(self.reward_clip))
            generated_reward = max(-limit, min(limit, generated_reward))
        return generated_reward, reward_terms

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

        generated_reward, reward_terms = self._safe_reward(obs, action, next_obs, original_reward, safe_info, progress)

        if isinstance(reward_terms, dict):
            safe_info["reward_terms"] = dict(reward_terms)
        elif "reward_terms" not in safe_info:
            safe_info["reward_terms"] = {}

        safe_info["original_env_reward"] = float(original_reward)
        safe_info["generated_reward"] = float(generated_reward)
        safe_info["training_progress"] = float(progress)
        safe_info["reward_error_count"] = int(self.reward_error_count)
        safe_info["reward_last_error"] = str(self.reward_last_error)
        safe_info["reward_terms"]["total_reward"] = float(generated_reward)

        self.last_obs = next_obs
        return next_obs, float(generated_reward), terminated, truncated, safe_info
