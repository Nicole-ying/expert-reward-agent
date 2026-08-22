#!/usr/bin/env python3
"""Budget-Matched Independent Generation Baseline (v2).

Generates K independent reward candidates per seed, using ONLY the raw task_spec
and masked_step_source — no environment card, no expert context, no RAG, no
role-based component budget. Each candidate is trained once and evaluated on the
original environment reward.

Usage:
  python baselines/run_independent_baseline.py --seeds 5 --samples 10
"""

import argparse, json, os, re, sys, time
from datetime import datetime
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


# ── Config ────────────────────────────────────────────────────────────────
ENV_ID = "LunarLander-v3"
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 4
EVAL_EPS = 20
EVAL_SEED_OFFSET = 10000
SOLVED_THRESHOLD = 200.0
REWARD_CLIP = 20.0

TASK_SPEC_PATH = "envs/env_001/task_spec_anonymized.yaml"
MASKED_STEP_PATH = "envs/env_001/masked_step_source.py"
SYSTEM_PROMPT_PATH = "prompts/02_reward_generator_prompt_independent_baseline.md"

OUTPUT_ROOT = "runs/env_001/budget_matched_independent_v2"
TENSORBOARD_DIR = "runs/env_001/tensorboard"

# PPO hyperparameters (matching CREATE paper config)
PPO_KWARGS = dict(
    policy="MlpPolicy",
    n_steps=1024,
    batch_size=64,
    gae_lambda=0.98,
    gamma=0.999,
    n_epochs=4,
    ent_coef=0.01,
    device="auto",
)


def extract_code(md: str) -> str:
    m = re.search(r"```python\s*(.*?)```", md, flags=re.S)
    if m:
        return m.group(1).strip()
    if "def compute_reward" in md:
        return md.strip()
    return ""


def make_reward_wrapper_class(reward_fn):
    """Create a wrapper class that replaces env reward with generated reward."""
    class _RewardWrapper(gym.Wrapper):
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
            try:
                gen_rew, _ = reward_fn(obs_before, action, next_obs, orig_rew, info, 0.0)
                gen_rew = float(gen_rew)
                if abs(gen_rew) > REWARD_CLIP:
                    gen_rew = max(-REWARD_CLIP, min(REWARD_CLIP, gen_rew))
            except Exception:
                gen_rew = 0.0
            return next_obs, gen_rew, terminated, truncated, info

    return _RewardWrapper


def load_reward_fn(reward_path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("reward_v1", str(Path(reward_path)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_reward


def generate_reward(client, system_prompt: str, task_spec: str, masked_step: str,
                    output_dir: Path) -> tuple[str, str]:
    """Call LLM to generate reward code. Returns (code, raw_llm_output)."""
    user_prompt = f"""# ANONYMIZED_TASK_SPEC

{task_spec}

# MASKED_STEP_SOURCE

```python
{masked_step}
```
"""
    # Save full input record
    full_input = f"# SYSTEM PROMPT\n\n{system_prompt}\n\n# USER PROMPT\n\n{user_prompt}"
    (output_dir / "llm_input_full.md").write_text(full_input, encoding="utf-8")

    response = client.chat(
        model="deepseek-chat",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.15,
        max_tokens=8192,
    )
    llm_output = response.strip()
    (output_dir / "llm_output.md").write_text(llm_output, encoding="utf-8")

    code = extract_code(llm_output)
    if not code:
        raise RuntimeError("Could not extract Python code from LLM output")
    return code, llm_output


def train_and_eval(reward_code: str, output_dir: Path, seed: int, sample_id: int) -> dict:
    """Train PPO with generated reward, evaluate on original env. Returns eval dict."""
    # Save reward code
    reward_path = output_dir / "reward_v1.py"
    reward_path.write_text(reward_code, encoding="utf-8")

    # Load reward function
    reward_fn = load_reward_fn(str(reward_path))
    RewardWrapper = make_reward_wrapper_class(reward_fn)

    # Build training envs
    def _make_wrapped():
        env = gym.make(ENV_ID)
        env.reset(seed=seed * 100 + sample_id)
        return Monitor(RewardWrapper(env))

    train_env = SubprocVecEnv([_make_wrapped for _ in range(N_ENVS)])

    # Train
    tensorboard_dir = Path(TENSORBOARD_DIR)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    model = PPO(
        env=train_env,
        verbose=1,
        seed=seed,
        tensorboard_log=str(tensorboard_dir),
        **PPO_KWARGS,
    )
    t0 = time.time()
    tb_name = f"indep_s{seed}_sp{sample_id}"
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=tb_name)
    train_sec = time.time() - t0

    model_path = output_dir / "model.zip"
    model.save(str(model_path))
    train_env.close()

    # Evaluate on ORIGINAL environment (no reward wrapper)
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

    mean_r = mean(episode_rewards)
    return {
        "seed": seed,
        "sample_id": sample_id,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_terminated": episode_terminated,
        "mean_eval_reward": mean_r,
        "mean_episode_length": mean(episode_lengths),
        "min_eval_reward": min(episode_rewards),
        "max_eval_reward": max(episode_rewards),
        "terminated_count": sum(1 for t in episode_terminated if t),
        "truncated_count": sum(1 for t in episode_terminated if not t),
        "solved": mean_r >= SOLVED_THRESHOLD,
        "train_sec": round(train_sec, 1),
        "train_min": round(train_sec / 60, 1),
        "reward_code_chars": len(reward_code),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="Number of independent seeds")
    ap.add_argument("--samples", type=int, default=10, help="Candidates per seed")
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="Print config and exit")
    ap.add_argument("--resume", action="store_true", help="Skip samples with existing model.zip")
    args = ap.parse_args()

    print("=" * 60)
    print("Budget-Matched Independent Generation Baseline (v2)")
    print(f"Env: {ENV_ID}  |  Seeds: {args.start_seed}..{args.start_seed + args.seeds - 1}")
    print(f"Samples/seed: {args.samples}  |  Total candidates: {args.seeds * args.samples}")
    print(f"Training steps/candidate: {TOTAL_TIMESTEPS:,}")
    print(f"Total training steps: {args.seeds * args.samples * TOTAL_TIMESTEPS:,}")
    print(f"Prompt: {SYSTEM_PROMPT_PATH}")
    print(f"Resume: {args.resume}")
    print("Input: task_spec + masked_step_source ONLY (no env card, no RAG)")
    print("=" * 60)

    if args.dry_run:
        return

    # Read inputs once
    system_prompt = Path(SYSTEM_PROMPT_PATH).read_text(encoding="utf-8")
    task_spec = Path(TASK_SPEC_PATH).read_text(encoding="utf-8")
    masked_step = Path(MASKED_STEP_PATH).read_text(encoding="utf-8")

    # Import LLM client
    from llm_clients.deepseek_client import DeepSeekClient
    client = DeepSeekClient()

    all_results = []
    # Load existing results when resuming (samples that have eval_result.json)
    if args.resume:
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            for sample in range(args.samples):
                eval_file = Path(OUTPUT_ROOT) / f"s{seed:02d}_c{sample:02d}" / "training" / "eval_result.json"
                if eval_file.exists():
                    try:
                        r = json.loads(eval_file.read_text(encoding="utf-8"))
                        all_results.append(r)
                    except Exception:
                        pass

    start_time = datetime.now()
    skipped = 0

    for seed in range(args.start_seed, args.start_seed + args.seeds):
        for sample in range(args.samples):
            sample_id = f"s{seed:02d}_c{sample:02d}"
            output_dir = Path(OUTPUT_ROOT) / sample_id / "training"

            # Resume: skip if model.zip already exists
            if args.resume and (output_dir / "model.zip").exists():
                print(f"[{sample_id}] SKIP (model.zip exists)")
                skipped += 1
                continue

            print(f"\n{'=' * 60}")
            print(f"Seed {seed}/{args.start_seed + args.seeds - 1}  Sample {sample}/{args.samples - 1}  [{sample_id}]")
            print(f"{'=' * 60}")

            output_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Generate reward (skip if LLM output already exists and resume)
            reward_path = output_dir / "reward_v1.py"
            if args.resume and reward_path.exists():
                print(f"[{sample_id}] Reusing existing reward_v1.py")
                code = reward_path.read_text(encoding="utf-8")
                gen_sec = 0
            else:
                print(f"[{sample_id}] Generating reward via LLM...")
                t0 = time.time()
                try:
                    code, llm_out = generate_reward(client, system_prompt, task_spec, masked_step, output_dir)
                    gen_sec = time.time() - t0
                    print(f"[{sample_id}] Generated ({len(code)} chars) in {gen_sec:.1f}s")
                except Exception as e:
                    print(f"[{sample_id}] LLM generation FAILED: {e}")
                    continue

            # Step 2: Train + Eval
            print(f"[{sample_id}] Training PPO ({TOTAL_TIMESTEPS:,} steps)...")
            try:
                result = train_and_eval(code, output_dir, seed, sample)
                result["sample_id"] = sample_id
                result["gen_sec"] = round(gen_sec, 1)
                result["reward_code"] = code
                all_results.append(result)

                # Save individual result immediately
                json.dump(result, open(output_dir / "eval_result.json", "w"),
                          indent=2, ensure_ascii=False)

                status = "[SOLVED]" if result["solved"] else "[FAILED]"
                print(f"[{sample_id}] {status}  mean={result['mean_eval_reward']:.2f}  "
                      f"range=[{result['min_eval_reward']:.1f}, {result['max_eval_reward']:.1f}]  "
                      f"train={result['train_min']:.1f}min  gen={gen_sec:.1f}s")
            except Exception as e:
                print(f"[{sample_id}] Training FAILED: {e}")
                import traceback
                traceback.print_exc()
                continue

    # ── Summary ──────────────────────────────────────────────────────────
    end_time = datetime.now()
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    scores = [r["mean_eval_reward"] for r in all_results]
    solved_count = sum(1 for r in all_results if r["solved"])
    total = len(all_results)

    print(f"Total completed: {total}")
    print(f"Skipped (already done): {skipped}")
    print(f"Solved (>= {SOLVED_THRESHOLD}): {solved_count}/{total}")
    if scores:
        print(f"Best: {max(scores):.3f}")
        print(f"Mean: {mean(scores):.3f}")
        print(f"Std:  {float(__import__('numpy').std(scores)):.3f}")
    print(f"Start: {start_time.isoformat()}")
    print(f"End:   {end_time.isoformat()}")

    # Write summary
    summary_dir = Path(OUTPUT_ROOT)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "# Budget-Matched Independent Generation (v2)",
        "",
        f"- num_seeds: {args.seeds}",
        f"- num_samples_per_seed: {args.samples}",
        f"- total_candidates: {total}",
        f"- env: {ENV_ID}",
        f"- training_steps_per_candidate: {TOTAL_TIMESTEPS:,}",
        f"- prompt: {SYSTEM_PROMPT_PATH}",
        f"- input: task_spec + masked_step_source ONLY",
        "",
        "## Results",
        "",
        "| seed | sample | score | solved (>=200) | terminated | truncated | length |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(all_results, key=lambda x: (x["seed"], x["sample_id"])):
        summary_lines.append(
            f"| {r['seed']} | {r['sample_id']} | {r['mean_eval_reward']:.3f} | "
            f"{'yes' if r['solved'] else 'no'} | "
            f"{r['terminated_count']} | {r['truncated_count']} | "
            f"{r['mean_episode_length']:.1f} |"
        )
    summary_lines.extend([
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| total_completed | {total} |",
        f"| solved | {solved_count}/{total} |",
    ])
    if scores:
        import numpy as np
        summary_lines.extend([
            f"| best | {max(scores):.3f} |",
            f"| mean | {mean(scores):.3f} |",
            f"| std | {float(np.std(scores)):.3f} |",
        ])
    (summary_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Write full results json
    json.dump(all_results, open(summary_dir / "results.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\nSummary written to: {summary_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
