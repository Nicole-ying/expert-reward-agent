#!/usr/bin/env python3
"""Ant-v4 PPO baseline — n_envs=4, n_steps=512, GPU, official reward, 3M steps, SubprocVecEnv."""

import json, sys, time
from datetime import datetime
from pathlib import Path
from functools import partial

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

ENV_ID = "Ant-v4"
TOTAL_TIMESTEPS = 3_000_000
FINAL_EVAL_EPS = 100
EVAL_SEED_BASE = 20000
NUM_SEEDS = 1
DEVICE = "cuda"
N_ENVS = 4
N_STEPS = 512
PREFIX = "baseline_ant_nenvs4_gpu_3M"

OUTPUT_DIR = Path(f"runs/env_005/{PREFIX}")


def _make_env(env_id, seed, rank, monitor_dir):
    """Module-level factory for SubprocVecEnv (must be picklable)."""
    env = gym.make(env_id)
    env.reset(seed=seed + rank)
    env = Monitor(env, str(Path(monitor_dir) / f"monitor_{rank}.csv"))
    return env


class Tee:
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
    def write(self, data):
        try: self.file.write(data)
        except ValueError: pass
        self.stdout.write(data)
    def flush(self):
        try: self.file.flush()
        except ValueError: pass
        self.stdout.flush()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(OUTPUT_DIR / "baseline.log")
    sys.stdout = tee

    start_time = datetime.now()
    print("=" * 60)
    print(f"Ant-v4 PPO Baseline — n_envs={N_ENVS}, n_steps={N_STEPS}, GPU, official reward")
    print(f"start: {start_time.isoformat()}")
    print(f"device: {DEVICE}  |  SubprocVecEnv  |  seeds: {NUM_SEEDS}")
    print(f"total_timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"rollout/update: {N_ENVS}×{N_STEPS}={N_ENVS*N_STEPS}")
    print("=" * 60)

    for seed in range(NUM_SEEDS):
        print(f"\n{'#'*60}\n SEED {seed}  (1/1)\n{'#'*60}")

        monitor_dir = OUTPUT_DIR / f"seed_{seed}" / "monitor"
        monitor_dir.mkdir(parents=True, exist_ok=True)
        tb_dir = OUTPUT_DIR / f"seed_{seed}" / "tensorboard"

        env_fns = [
            partial(_make_env, ENV_ID, seed, rank, str(monitor_dir))
            for rank in range(N_ENVS)
        ]
        train_env = SubprocVecEnv(env_fns)

        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=3e-4, n_steps=N_STEPS, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                               activation_fn=torch.nn.Tanh, ortho_init=True),
            device=DEVICE, seed=seed, verbose=1,
            tensorboard_log=str(tb_dir),
        )

        t0 = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        train_sec = time.time() - t0

        model_path = OUTPUT_DIR / f"seed_{seed}" / "model.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        train_env.close()

        eval_env = Monitor(gym.make(ENV_ID))
        rewards, lengths = [], []
        for ep in range(FINAL_EVAL_EPS):
            obs, _ = eval_env.reset(seed=EVAL_SEED_BASE + ep)
            done, truncated, ep_r, ep_l = False, False, 0.0, 0
            while not done and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = eval_env.step(action)
                ep_r += float(reward); ep_l += 1
            rewards.append(ep_r); lengths.append(ep_l)
        eval_env.close()

        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards))
        print(f"SEED {seed} FINAL: mean={mean_r:.2f} +- {std_r:.2f}  |  {train_sec/60:.1f} min")

    json.dump({
        "experiment": f"Ant-v4 PPO baseline — n_envs={N_ENVS}, GPU, official reward",
        "env_id": ENV_ID, "device": DEVICE, "n_envs": N_ENVS,
        "start": start_time.isoformat(),
        "seed_0": {"mean_reward": mean_r, "std_reward": std_r, "train_min": train_sec/60},
    }, open(OUTPUT_DIR / "baseline_results.json", "w"), indent=2)

    tee.file.close()


if __name__ == "__main__":
    main()
