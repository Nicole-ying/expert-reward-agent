#!/usr/bin/env python3
"""LunarLander-v3 PPO baseline — SB3 defaults, CPU, verbose=1."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# ── config ──────────────────────────────────────────────────────────────
ENV_ID = "LunarLander-v3"
TOTAL_TIMESTEPS = 1_000_000
FINAL_EVAL_EPS = 100
EVAL_SEED_BASE = 20000
SEED = 0
DEVICE = "cpu"

OUTPUT_DIR = Path("runs/env_001/baselines/ppo_default_cpu")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "training.log"

# Tee: write to both file and stdout
class Tee:
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

tee = Tee(LOG_FILE)
sys.stdout = tee

start_time = datetime.now()
print("=" * 60)
print(f"LunarLander-v3 PPO Baseline — SB3 defaults, CPU")
print(f"start: {start_time.isoformat()}")
print("=" * 60)
print(f"env          : {ENV_ID}")
print(f"total steps  : {TOTAL_TIMESTEPS:,}")
print(f"final eval   : {FINAL_EVAL_EPS} episodes")
print(f"seed         : {SEED}")
print(f"device       : {DEVICE}")
print(f"output       : {OUTPUT_DIR}")
print()

# ── train ───────────────────────────────────────────────────────────────
print("Creating env + PPO ...")
train_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID))])

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                       activation_fn=torch.nn.Tanh, ortho_init=True),
    device=DEVICE,
    seed=SEED,
    verbose=1,
)

print(f"Training {TOTAL_TIMESTEPS:,} steps ...\n")
t0 = time.time()
model.learn(total_timesteps=TOTAL_TIMESTEPS)
train_sec = time.time() - t0

print(f"\nTraining done in {train_sec/60:.1f} min")

# ── save model ──────────────────────────────────────────────────────────
model_path = OUTPUT_DIR / "ppo_lunarlander_baseline.zip"
model.save(str(model_path))
print(f"Model saved → {model_path} ({model_path.stat().st_size/1e6:.1f} MB)")

train_env.close()

# ── final evaluation (100 episodes) ─────────────────────────────────────
print(f"\nFinal eval: {FINAL_EVAL_EPS} episodes (deterministic) ...")
eval_env = Monitor(gym.make(ENV_ID))
all_rewards, all_lengths = [], []

for ep in range(FINAL_EVAL_EPS):
    obs, _ = eval_env.reset(seed=EVAL_SEED_BASE + ep)
    done, truncated, ep_r, ep_l = False, False, 0.0, 0
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = eval_env.step(action)
        ep_r += float(reward)
        ep_l += 1
    all_rewards.append(ep_r)
    all_lengths.append(ep_l)

eval_env.close()

mean_r = float(np.mean(all_rewards))
std_r = float(np.std(all_rewards))
min_r = float(np.min(all_rewards))
max_r = float(np.max(all_rewards))
mean_len = float(np.mean(all_lengths))
solved = mean_r >= 200.0

end_time = datetime.now()

# ── results ─────────────────────────────────────────────────────────────
results = {
    "experiment": "LunarLander-v3 PPO baseline (SB3 defaults, CPU)",
    "env_id": ENV_ID,
    "seed": SEED,
    "device": DEVICE,
    "total_timesteps": TOTAL_TIMESTEPS,
    "start_time": start_time.isoformat(),
    "end_time": end_time.isoformat(),
    "train_seconds": round(train_sec, 1),
    "final_eval": {
        "n_episodes": FINAL_EVAL_EPS,
        "mean_reward": mean_r,
        "std_reward": std_r,
        "min_reward": min_r,
        "max_reward": max_r,
        "mean_ep_len": mean_len,
        "solved": bool(solved),
    },
    "rewards": all_rewards,
    "lengths": all_lengths,
}

json_path = OUTPUT_DIR / "baseline_results.json"
json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ── summary ─────────────────────────────────────────────────────────────
summary_path = OUTPUT_DIR / "baseline_summary.md"
summary = f"""# LunarLander-v3 PPO Baseline (SB3 Defaults, CPU)

- **env**: {ENV_ID}
- **seed**: {SEED}
- **device**: {DEVICE}
- **total_timesteps**: {TOTAL_TIMESTEPS:,}
- **start**: {start_time.isoformat()}
- **end**: {end_time.isoformat()}
- **train_time**: {train_sec/60:.1f} min

## PPO Parameters (SB3 defaults)

| param | value |
|-------|-------|
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.0 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| lr | 3e-4 |
| net_arch | pi=[64,64], vf=[64,64] |
| activation | Tanh |
| ortho_init | True |

## Final Evaluation ({FINAL_EVAL_EPS} episodes, deterministic)

| metric | value |
|--------|-------|
| **mean_reward** | **{mean_r:.2f}** |
| **std_reward** | **{std_r:.2f}** |
| min | {min_r:.2f} |
| max | {max_r:.2f} |
| mean_ep_len | {mean_len:.1f} |
| solved (≥200) | **{'✅ YES' if solved else '❌ NO'}** |

## Model

- `{model_path}` ({model_path.stat().st_size/1e6:.1f} MB)
"""
summary_path.write_text(summary, encoding="utf-8")

print(f"\n{'='*60}")
print(f"FINAL: mean={mean_r:.2f} ± {std_r:.2f}  |  [{min_r:.0f}, {max_r:.0f}]")
print(f"ep_len={mean_len:.1f}  |  solved={'YES ✅' if solved else 'NO ❌'}")
print(f"end: {end_time.isoformat()}")
print(f"train: {train_sec/60:.1f} min")
print(f"\nSaved: model → {model_path}")
print(f"       json  → {json_path}")
print(f"       md    → {summary_path}")
print(f"       log   → {LOG_FILE}")

tee.file.close()
