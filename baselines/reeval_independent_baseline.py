#!/usr/bin/env python3
"""Re-evaluate completed independent baseline samples.

For every sample directory under OUTPUT_ROOT that has model.zip but no
eval_result.json, load the model and reward function, run eval episodes on
the original environment, and write eval_result.json.
"""
import json, sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

ENV_ID = "LunarLander-v3"
EVAL_EPS = 20
EVAL_SEED_OFFSET = 10000
SOLVED_THRESHOLD = 200.0
REWARD_CLIP = 20.0
OUTPUT_ROOT = "runs/env_001/budget_matched_independent_v2"


def load_reward_fn(reward_path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("reward_v1", str(Path(reward_path)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_reward


def main():
    root = Path(OUTPUT_ROOT)
    dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    done = 0

    for d in dirs:
        training_dir = d / "training"
        model_path = training_dir / "model.zip"
        eval_path = training_dir / "eval_result.json"
        reward_path = training_dir / "reward_v1.py"

        if not model_path.exists():
            print(f"[{d.name}] SKIP: no model.zip")
            continue
        if eval_path.exists():
            print(f"[{d.name}] SKIP: eval_result.json already exists")
            continue
        if not reward_path.exists():
            print(f"[{d.name}] SKIP: no reward_v1.py")
            continue

        print(f"[{d.name}] Evaluating...", end=" ", flush=True)

        try:
            # Extract seed and sample from directory name like "s01_c03"
            seed = int(d.name[1:3])
        except ValueError:
            seed = 0

        try:
            reward_fn = load_reward_fn(str(reward_path))
            model = PPO.load(str(model_path))

            eval_env = Monitor(gym.make(ENV_ID))
            episode_rewards = []
            episode_lengths = []
            episode_terminated = []

            for ep in range(EVAL_EPS):
                eval_seed = EVAL_SEED_OFFSET + ep
                obs, _ = eval_env.reset(seed=eval_seed)
                done, ep_r, ep_l, was_terminated = False, 0.0, 0, False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    ep_r += float(reward)
                    ep_l += 1
                    done = bool(terminated or truncated)
                    if terminated:
                        was_terminated = True
                episode_rewards.append(ep_r)
                episode_lengths.append(ep_l)
                episode_terminated.append(was_terminated)
            eval_env.close()

            result = {
                "seed": seed,
                "sample_id": d.name,
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                "episode_terminated": episode_terminated,
                "mean_eval_reward": mean(episode_rewards),
                "mean_episode_length": mean(episode_lengths),
                "min_eval_reward": min(episode_rewards),
                "max_eval_reward": max(episode_rewards),
                "terminated_count": sum(1 for t in episode_terminated if t),
                "truncated_count": sum(1 for t in episode_terminated if not t),
                "solved": mean(episode_rewards) >= SOLVED_THRESHOLD,
            }
            json.dump(result, open(eval_path, "w"), indent=2, ensure_ascii=False)

            status = "[SOLVED]" if result["solved"] else "[FAILED]"
            print(f"{status} mean={result['mean_eval_reward']:.2f} "
                  f"range=[{result['min_eval_reward']:.1f}, {result['max_eval_reward']:.1f}] "
                  f"len={result['mean_episode_length']:.1f}")
            done += 1

        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDone. Re-evaluated {done} samples.")


if __name__ == "__main__":
    main()
