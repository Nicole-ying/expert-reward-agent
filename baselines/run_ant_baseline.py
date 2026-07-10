#!/usr/bin/env python3
"""Ant-v4 PPO baseline — RL Zoo / OpenAI params, official reward, 1 env, CPU, 3 seeds."""

import json, sys, time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ENV_ID = "Ant-v4"
TOTAL_TIMESTEPS = 1_000_000
FINAL_EVAL_EPS = 100
EVAL_SEED_BASE = 20000
NUM_SEEDS = 3
DEVICE = "cpu"
PREFIX = "baseline_ant_sb3_ppo_v1"

OUTPUT_DIR = Path(f"runs/env_005/{PREFIX}")


def _make_env():
    return Monitor(gym.make(ENV_ID))


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
    print(f"Ant-v4 PPO Baseline — RL Zoo / OpenAI PPO params, official reward")
    print(f"start: {start_time.isoformat()}")
    print(f"device: {DEVICE}  |  n_envs: 1 (DummyVecEnv)  |  seeds: {NUM_SEEDS}")
    print(f"total_timesteps: {TOTAL_TIMESTEPS:,}")
    print("=" * 60)
    print()
    print("PPO params: n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95")
    print("            lr=3e-4, clip=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5")
    print("            net_arch=[64,64], Tanh, ortho_init")
    print()

    all_results = {}

    for seed in range(NUM_SEEDS):
        print(f"\n{'#'*60}")
        print(f" SEED {seed}  ({seed+1}/{NUM_SEEDS})")
        print(f"{'#'*60}")

        train_env = DummyVecEnv([_make_env])

        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                               activation_fn=torch.nn.Tanh, ortho_init=True),
            device=DEVICE, seed=seed, verbose=1,
        )

        t0 = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        train_sec = time.time() - t0

        model_path = OUTPUT_DIR / f"seed_{seed}" / "model.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        print(f"Model saved -> {model_path}")

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

        all_results[f"seed_{seed}"] = {
            "mean_reward": mean_r, "std_reward": std_r,
            "min": float(np.min(rewards)), "max": float(np.max(rewards)),
            "mean_ep_len": float(np.mean(lengths)),
            "train_min": round(train_sec / 60, 1),
        }

        print(f"SEED {seed} FINAL: mean={mean_r:.2f} +- {std_r:.2f}  |  {train_sec/60:.1f} min")

    end_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"SUMMARY ({NUM_SEEDS} seeds)")
    print(f"{'='*60}")
    print(f"{'seed':<8} {'mean':>10} {'std':>10} {'time':>10}")
    for seed in range(NUM_SEEDS):
        r = all_results[f"seed_{seed}"]
        print(f"{seed:<8} {r['mean_reward']:>10.1f} {r['std_reward']:>10.1f} {r['train_min']:>9.1f}m")

    means = [all_results[f"seed_{s}"]["mean_reward"] for s in range(NUM_SEEDS)]
    print(f"\nOverall: {np.mean(means):.1f} +- {np.std(means):.1f}")
    print(f"start: {start_time.isoformat()}")
    print(f"end:   {end_time.isoformat()}")

    json.dump({
        "experiment": "Ant-v4 PPO baseline (RL Zoo / OpenAI params, official reward, n_envs=1)",
        "env_id": ENV_ID, "device": DEVICE, "n_envs": 1,
        "start": start_time.isoformat(), "end": end_time.isoformat(),
        "seeds": all_results,
        "overall_mean": float(np.mean(means)),
        "overall_std": float(np.std(means)),
    }, open(OUTPUT_DIR / "baseline_results.json", "w"), indent=2, ensure_ascii=False)

    tee.file.close()


if __name__ == "__main__":
    main()
