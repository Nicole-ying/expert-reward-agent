#!/usr/bin/env python3
"""Ablation v2: no environment card, no expert schema.

Feeds ONLY raw task_spec + masked_step_source to the reward generator LLM.
No environment card. No formula operator library. Just the system prompt.
"""

import json, os, re, sys, time, subprocess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# ── Config ────────────────────────────────────────────────────────────────
ENV_ID = "BipedalWalker-v3"
TOTAL_TIMESTEPS = 1_000_000
FINAL_EVAL_EPS = 100
EVAL_SEED_BASE = 20000
SEED = 0
DEVICE = "cpu"
PREFIX = "ablation_direct_no_expert_v3_chat"

TASK_SPEC_PATH = "envs/env_002/task_spec_anonymized.yaml"
MASKED_STEP_PATH = "envs/env_002/masked_step_source.py"
SYSTEM_PROMPT_PATH = "prompts/02_reward_generator_prompt_direct.md"

OUTPUT_DIR = Path(f"runs/env_002/{PREFIX}")


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


def extract_code(md):
    m = re.search(r"```python\s*(.*?)```", md, flags=re.S)
    if m:
        return m.group(1).strip()
    if "def compute_reward" in md:
        return md.strip()
    return ""


def _make_env():
    return Monitor(gym.make(ENV_ID))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Tee to capture ALL stdout (including subprocess) ──────────────────
    log_path = OUTPUT_DIR / "experiment.log"
    # We can't use Tee for subprocess output easily, so write directly
    log_f = open(log_path, "w", encoding="utf-8", buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    start_time = datetime.now()
    log("=" * 60)
    log("Ablation v2: No environment card, no expert schema")
    log(f"start: {start_time.isoformat()}")
    log(f"env: {ENV_ID}  |  seed: {SEED}  |  steps: {TOTAL_TIMESTEPS:,}")
    log("Input: task_spec + masked_step_source ONLY")
    log("=" * 60)

    # ── Step 1: Generate reward via LLM ────────────────────────────────────
    log("\n--- Step 1: Generate reward (LLM) ---")
    t0 = time.time()

    from llm_clients.deepseek_client import DeepSeekClient

    system_prompt = Path(SYSTEM_PROMPT_PATH).read_text(encoding="utf-8")
    task_spec = Path(TASK_SPEC_PATH).read_text(encoding="utf-8")
    masked_step = Path(MASKED_STEP_PATH).read_text(encoding="utf-8")

    user_prompt = f"""# ANONYMIZED_TASK_SPEC

{task_spec}

# MASKED_STEP_SOURCE

```python
{masked_step}
```
"""

    # ── Save full LLM input record ─────────────────────────────────────────
    full_input = f"""# SYSTEM PROMPT

{system_prompt}

# USER PROMPT

{user_prompt}
"""
    (OUTPUT_DIR / "llm_input_full.md").write_text(full_input, encoding="utf-8")
    log(f"LLM input saved: llm_input_full.md ({len(full_input)} chars)")

    client = DeepSeekClient()
    response = client.chat(
        model="deepseek-chat",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.15,
        max_tokens=8192,
    )
    llm_output = response.strip()
    (OUTPUT_DIR / "llm_output.md").write_text(llm_output, encoding="utf-8")

    code = extract_code(llm_output)
    if not code:
        log("ERROR: Could not extract code from LLM output")
        log_f.close()
        sys.exit(1)

    reward_path = OUTPUT_DIR / "reward_v1.py"
    reward_path.write_text(code, encoding="utf-8")
    gen_sec = time.time() - t0
    log(f"Generated ({len(code)} chars) in {gen_sec:.1f}s")
    log(f"Reward code:\n{code}")

    # ── Step 2: Train PPO ─────────────────────────────────────────────────
    log(f"\n--- Step 2: Training PPO ({TOTAL_TIMESTEPS:,} steps, n_envs=8) ---")
    t0 = time.time()

    from stable_baselines3.common.vec_env import SubprocVecEnv
    train_env = SubprocVecEnv([_make_env for _ in range(8)])

    # Load reward module
    import importlib.util
    spec = importlib.util.spec_from_file_location("reward_v1", str(reward_path))
    reward_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward_mod)

    # Wrap env with custom reward
    class RewardWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self._last_obs = None
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            self._last_obs = obs
            return obs, info
        def step(self, action):
            next_obs, orig_rew, terminated, truncated, info = self.env.step(action)
            obs_before = self._last_obs
            self._last_obs = next_obs
            rew, _ = reward_mod.compute_reward(obs_before, action, next_obs, orig_rew, info, 0.0)
            return next_obs, float(rew), terminated, truncated, info

    def _make_wrapped():
        return Monitor(RewardWrapper(gym.make(ENV_ID)))

    train_env = SubprocVecEnv([_make_wrapped for _ in range(8)])

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4, n_steps=2048, batch_size=128,
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                           activation_fn=torch.nn.Tanh, ortho_init=True),
        device=DEVICE, seed=SEED, verbose=1,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    train_sec = time.time() - t0

    model_path = OUTPUT_DIR / "model.zip"
    model.save(str(model_path))
    log(f"Model saved -> {model_path} ({model_path.stat().st_size/1e6:.1f} MB)")
    train_env.close()

    # ── Step 3: Evaluate 100 episodes (official reward) ────────────────────
    log(f"\n--- Step 3: Evaluating ({FINAL_EVAL_EPS} episodes, deterministic) ---")

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
    min_r = float(np.min(rewards))
    max_r = float(np.max(rewards))
    solved = mean_r >= 300.0

    end_time = datetime.now()

    log(f"\n{'='*60}")
    log(f"EVAL ({FINAL_EVAL_EPS} eps): mean={mean_r:.2f} +- {std_r:.2f}  |  min={min_r:.1f}  max={max_r:.1f}")
    log(f"Solved (>=300): {'YES' if solved else 'NO'}")
    log(f"LLM gen: {gen_sec:.1f}s  |  Train: {train_sec/60:.1f} min")
    log(f"start: {start_time.isoformat()}")
    log(f"end:   {end_time.isoformat()}")

    # ── Step 4: Save results ──────────────────────────────────────────────
    results = {
        "experiment": "Ablation v2: no environment card, no expert schema",
        "env_id": ENV_ID, "seed": SEED, "device": DEVICE,
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_envs": 8,
        "start": start_time.isoformat(), "end": end_time.isoformat(),
        "llm_gen_sec": round(gen_sec, 1),
        "train_sec": round(train_sec, 1),
        "train_min": round(train_sec / 60, 1),
        "reward_code_chars": len(code),
        "reward_code": code,
        "final_eval": {
            "n_episodes": FINAL_EVAL_EPS,
            "mean_reward": mean_r, "std_reward": std_r,
            "min": min_r, "max": max_r,
            "mean_ep_len": float(np.mean(lengths)),
            "solved": solved,
        },
        "rewards": rewards,
        "lengths": lengths,
    }
    json.dump(results, open(OUTPUT_DIR / "results.json", "w"), indent=2, ensure_ascii=False)
    log(f"\nSaved: {OUTPUT_DIR / 'results.json'}")

    log_f.close()


if __name__ == "__main__":
    main()
