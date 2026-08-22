#!/usr/bin/env python3
"""Train subagent iter_03 reward (eval=1840) for 3M steps, eval on official reward."""

import sys; sys.path.insert(0, "/home/utseus22/expert-reward-agent")
import json, time
from datetime import datetime
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from training.reward_wrapper import RewardOverrideWrapper, load_reward_function

ENV_ID = "Ant-v4"
TOTAL_TIMESTEPS = 3_000_000
FINAL_EVAL_EPS = 100
EVAL_SEED_BASE = 20000
NUM_SEEDS = 3
DEVICE = "cpu"
PREFIX = "subagent_iter03_3M"

REWARD_PATH = Path("runs/env_005/paper_ant_subagent_20260728/seed_0/iter_03/generation/reward_v3.py")
OUTPUT_DIR = Path(f"runs/env_005/{PREFIX}")


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
    if not REWARD_PATH.exists():
        print(f"ERROR: {REWARD_PATH} not found")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(OUTPUT_DIR / "baseline.log")
    sys.stdout = tee

    start_time = datetime.now()
    print("=" * 60)
    print(f"Ant-v4 PPO 3M — subagent iter_03 reward (eval=1840)")
    print(f"reward: {REWARD_PATH}")
    print(f"start: {start_time.isoformat()}")
    print(f"device: {DEVICE}  |  n_envs: 1  |  seeds: {NUM_SEEDS}")
    print(f"total_timesteps: {TOTAL_TIMESTEPS:,}")
    print("=" * 60)

    reward_fn = load_reward_function(str(REWARD_PATH))
    all_results = {}

    for seed in range(NUM_SEEDS):
        print(f"\n{'#'*60}\n SEED {seed}  ({seed+1}/{NUM_SEEDS})\n{'#'*60}")

        train_env = DummyVecEnv([lambda: RewardOverrideWrapper(
            Monitor(gym.make(ENV_ID)), reward_fn,
            max_training_steps_for_progress=TOTAL_TIMESTEPS)])

        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                               activation_fn=torch.nn.Tanh, ortho_init=True),
            device=DEVICE, seed=seed, verbose=1,
            tensorboard_log=str(OUTPUT_DIR / f"seed_{seed}" / "tensorboard"),
        )

        t0 = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        train_sec = time.time() - t0

        model_path = OUTPUT_DIR / f"seed_{seed}" / "model.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        train_env.close()

        # Eval with OFFICIAL reward
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
    means = [all_results[f"seed_{s}"]["mean_reward"] for s in range(NUM_SEEDS)]
    print(f"\nOverall: {np.mean(means):.1f} +- {np.std(means):.1f}")

    json.dump({
        "experiment": f"Ant-v4 PPO 3M — subagent iter_03 reward, official eval",
        "env_id": ENV_ID, "device": DEVICE, "n_envs": 1,
        "reward_path": str(REWARD_PATH),
        "start": start_time.isoformat(), "end": end_time.isoformat(),
        "seeds": all_results,
        "overall_mean": float(np.mean(means)),
        "overall_std": float(np.std(means)),
    }, open(OUTPUT_DIR / "baseline_results.json", "w"), indent=2, ensure_ascii=False)

    tee.file.close()


if __name__ == "__main__":
    main()
