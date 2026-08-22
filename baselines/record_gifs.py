#!/usr/bin/env python3
"""Record GIFs of trained Ant policies walking on official reward."""
import os, sys, imageio
os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

MODELS = [
    ("baseline_1M_2569",    "runs/env_005/baseline_ant_1M_retrain_v2/seed_0/model.zip", None),
    ("baseline_3M_s0_2728", "runs/env_005/baseline_ant_sb3_ppo_3M/seed_0/model.zip", None),
    ("baseline_3M_s2_2912", "runs/env_005/baseline_ant_sb3_ppo_3M/seed_2/model.zip", None),
    ("v7_iter08_1M",        "runs/env_005/paper_ant_v7/seed_0/iter_08/training/model.zip",
                            "runs/env_005/paper_ant_v7/seed_0/iter_08/training/vecnormalize.pkl"),
    ("subagent_iter03_1M",  "runs/env_005/paper_ant_subagent_20260728/seed_0/iter_03/training/model.zip",
                            "runs/env_005/paper_ant_subagent_20260728/seed_0/iter_03/training/vecnormalize.pkl"),
]

OUT_DIR = "logs/gifs"
os.makedirs(OUT_DIR, exist_ok=True)

for name, model_path, vn_path in MODELS:
    print(f"Recording {name}...", flush=True)
    if not os.path.exists(model_path):
        print(f"  SKIP: model not found", flush=True)
        continue

    model = PPO.load(model_path, device="cpu")

    # Load VecNormalize if available (framework-trained models need this)
    obs_norm = None
    if vn_path and os.path.exists(vn_path):
        obs_norm = VecNormalize.load(vn_path, venv=DummyVecEnv([lambda: gym.make("Ant-v4")]))
        print(f"  VecNormalize loaded", flush=True)

    env = gym.make("Ant-v4", render_mode="rgb_array")

    frames = []
    obs, _ = env.reset(seed=42)
    total_reward = 0

    for _ in range(1000):
        # Apply VecNormalize to observation if needed
        if obs_norm is not None:
            policy_obs = obs_norm.normalize_obs(obs[None, ...])[0]
        else:
            policy_obs = obs

        action, _ = model.predict(policy_obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        frames.append(env.render())
        if terminated or truncated:
            break

    env.close()

    out_path = f"{OUT_DIR}/{name}.gif"
    imageio.mimsave(out_path, frames, fps=30, loop=0)
    print(f"  -> {out_path}  frames={len(frames)}  official_reward={total_reward:.0f}", flush=True)

print("\nDone!", flush=True)
