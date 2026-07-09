#!/usr/bin/env python3
"""Official reward baseline — LunarLander-tuned PPO params, GPU 0, 5 seeds."""

import json, sys, time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ENV_ID = "LunarLander-v3"
TOTAL_TIMESTEPS = 1_000_000
FINAL_EVAL_EPS = 100
EVAL_SEED_BASE = 20000
NUM_SEEDS = 5
DEVICE = "cuda:0"
PREFIX = "baseline_lander_tuned_gpu0_v1"

OUTPUT_DIR = Path(f"runs/env_001/{PREFIX}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

tee = Tee(OUTPUT_DIR / "baseline.log")
sys.stdout = tee

start_time = datetime.now()
print("=" * 60)
print(f"LunarLander-v3 Official Reward Baseline — Tuned PPO, GPU 0")
print(f"start: {start_time.isoformat()}")
print(f"device: {DEVICE}  |  seeds: {NUM_SEEDS}")
print("=" * 60)
print(f"n_envs=4  n_steps=1024  gamma=0.999  gae_lambda=0.98  n_epochs=4  ent_coef=0.01")
print()

all_results = {}

for seed in range(NUM_SEEDS):
    print(f"\n{'#'*60}")
    print(f" SEED {seed}  ({seed+1}/{NUM_SEEDS})")
    print(f"{'#'*60}")

    train_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID)) for _ in range(4)])

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4, n_steps=1024, batch_size=64,
        n_epochs=4, gamma=0.999, gae_lambda=0.98,
        clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[64,64], vf=[64,64]),
                           activation_fn=torch.nn.Tanh, ortho_init=True),
        device=DEVICE, seed=seed, verbose=1,
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    train_sec = time.time() - t0

    model_path = OUTPUT_DIR / f"seed_{seed}" / "model.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"Model saved → {model_path}")

    train_env.close()

    # Final eval
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
    solved = mean_r >= 200.0

    all_results[f"seed_{seed}"] = {
        "mean_reward": mean_r, "std_reward": std_r,
        "min": float(np.min(rewards)), "max": float(np.max(rewards)),
        "mean_ep_len": float(np.mean(lengths)), "solved": solved,
        "train_min": round(train_sec/60, 1),
    }

    print(f"SEED {seed} FINAL: mean={mean_r:.2f} ± {std_r:.2f}  |  solved={'YES' if solved else 'NO'}  |  {train_sec/60:.1f} min")

# Summary
end_time = datetime.now()
print(f"\n{'='*60}")
print(f"SUMMARY (5 seeds)")
print(f"{'='*60}")
print(f"{'seed':<8} {'mean':>8} {'std':>8} {'solved':>8} {'time':>8}")
for seed in range(NUM_SEEDS):
    r = all_results[f"seed_{seed}"]
    print(f"{seed:<8} {r['mean_reward']:>8.1f} {r['std_reward']:>8.1f} {'YES' if r['solved'] else 'NO':>8} {r['train_min']:>7.1f}m")

means = [all_results[f"seed_{s}"]["mean_reward"] for s in range(NUM_SEEDS)]
print(f"\nOverall: {np.mean(means):.1f} ± {np.std(means):.1f}  |  solved {sum(1 for m in means if m>=200)}/{NUM_SEEDS}")
print(f"start: {start_time.isoformat()}")
print(f"end:   {end_time.isoformat()}")

json.dump({
    "experiment": "LunarLander-v3 official reward baseline (tuned PPO, GPU 0)",
    "device": DEVICE, "ppo": "tuned LunarLander params",
    "start": start_time.isoformat(), "end": end_time.isoformat(),
    "seeds": all_results,
    "overall_mean": float(np.mean(means)),
    "overall_std": float(np.std(means)),
    "solved_count": sum(1 for m in means if m>=200),
}, open(OUTPUT_DIR / "baseline_results.json", "w"), indent=2, ensure_ascii=False)

tee.file.close()
