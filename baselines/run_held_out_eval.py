#!/usr/bin/env python3
"""Unified Held-Out Evaluation for all paper experiments.

For each experiment, loads the BEST model+reward per seed, evaluates on
100 unified held-out episodes using the ORIGINAL environment reward.
All methods use identical seeds for fair comparison.

Output per experiment:
  - held_out_results.json : per-episode rewards + summary stats
  - held_out_summary.md   : human-readable table
"""

import json, sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

ENV_ID = "LunarLander-v3"
HELD_OUT_EPS = 100
HELD_OUT_SEED_START = 50000  # Far from dev seeds (10000-10019)
SOLVED_THRESHOLD = 200.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "runs" / "env_001"

# ── Experiment definitions ────────────────────────────────────────────────
EXPERIMENTS = {
    "CREATE": {
        "dir": "paper_v4",
        "type": "iterative",  # best_training_summary.json has model_path
    },
    "CoarseFeedback": {
        "dir": "ablation_eureka_feedback_v4",
        "type": "iterative",
    },
    "Unconstrained": {
        "dir": "ablation_unconstrained_v4",
        "type": "iterative",
    },
    "IndependentGen": {
        "dir": "budget_matched_independent_v2",
        "type": "independent",  # needs per-seed best from 10 candidates
    },
}


def load_reward_fn(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("reward", str(Path(path)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_reward


def evaluate_model(model, env_id, seeds, reward_fn=None):
    """Run eval episodes on ORIGINAL env (no reward wrapper).
    Returns list of (total_reward, length, terminated) per episode."""
    results = []
    for ep_seed in seeds:
        env = gym.make(env_id)
        obs, _ = env.reset(seed=ep_seed)
        done, ep_r, ep_l, was_terminated = False, 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_r += float(reward)
            ep_l += 1
            done = bool(terminated or truncated)
            if terminated:
                was_terminated = True
        env.close()
        results.append({
            "seed": ep_seed,
            "reward": ep_r,
            "length": ep_l,
            "terminated": was_terminated,
        })
    return results


def eval_iterative_experiment(exp_name, exp_cfg, seeds, output_dir):
    """Evaluate best model from iterative experiments (CREATE, ablations)."""
    exp_dir = RUNS_ROOT / exp_cfg["dir"]
    all_results = {}

    for seed_dir in sorted(exp_dir.glob("seed_*")):
        seed_name = seed_dir.name
        summary_path = seed_dir / "best" / "best_training_summary.json"
        if not summary_path.exists():
            print(f"  {seed_name}: no best_training_summary.json, SKIP")
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model_rel = summary.get("model_path", "")
        reward_rel = summary.get("reward_path", "")
        dev_score = summary.get("external_eval", {}).get("mean_eval_reward", None)

        model_path = PROJECT_ROOT / model_rel
        reward_path = PROJECT_ROOT / reward_rel

        if not model_path.exists():
            print(f"  {seed_name}: model MISSING at {model_rel}, SKIP")
            continue

        dev_str = f"{dev_score:.2f}" if dev_score is not None else "?"
        print(f"  {seed_name}: loading model (dev={dev_str})...", end=" ", flush=True)

        try:
            model = PPO.load(str(model_path))
            reward_fn = load_reward_fn(str(reward_path)) if reward_path.exists() else None
            ep_results = evaluate_model(model, ENV_ID, seeds, reward_fn)
            rewards = [r["reward"] for r in ep_results]
            lengths = [r["length"] for r in ep_results]
            terminated = sum(1 for r in ep_results if r["terminated"])

            seed_result = {
                "seed": seed_name,
                "dev_score": dev_score,
                "model_path": model_rel,
                "reward_path": reward_rel,
                "held_out_mean": mean(rewards),
                "held_out_std": stdev(rewards) if len(rewards) > 1 else 0,
                "held_out_min": min(rewards),
                "held_out_max": max(rewards),
                "held_out_mean_len": mean(lengths),
                "held_out_terminated": terminated,
                "held_out_truncated": HELD_OUT_EPS - terminated,
                "solved": mean(rewards) >= SOLVED_THRESHOLD,
                "episodes": ep_results,
            }
            all_results[seed_name] = seed_result

            status = "[SOLVED]" if seed_result["solved"] else "       "
            print(f"{status} held-out={mean(rewards):.2f}+-{stdev(rewards) if len(rewards) > 1 else 0:.2f}  "
                  f"term={terminated}/{HELD_OUT_EPS}  len={mean(lengths):.1f}")
            del model  # free memory
        except Exception as e:
            print(f"ERROR: {e}")

    # Write
    output_file = output_dir / f"held_out_{exp_name}.json"
    json.dump(all_results, open(output_file, "w"), indent=2, ensure_ascii=False)
    write_summary_md(all_results, exp_name, output_dir)
    return all_results


def eval_independent_experiment(exp_name, exp_cfg, seeds, output_dir):
    """For independent generation: pick best candidate per seed from 10 samples."""
    exp_dir = RUNS_ROOT / exp_cfg["dir"]
    all_results = {}

    # Group samples by seed
    seed_candidates = {}
    for sample_dir in sorted(exp_dir.glob("s*_c*")):
        name = sample_dir.name  # e.g., "s00_c05"
        seed = int(name[1:3])
        eval_file = sample_dir / "training" / "eval_result.json"
        if eval_file.exists():
            r = json.loads(eval_file.read_text(encoding="utf-8"))
            seed_candidates.setdefault(seed, []).append((name, r))

    if not seed_candidates:
        print("  No eval results found yet. Waiting for training to complete.")
        return {}

    for seed in sorted(seed_candidates.keys()):
        candidates = seed_candidates[seed]
        # Pick best by dev score
        best_name, best_r = max(candidates, key=lambda x: x[1]["mean_eval_reward"])
        model_path = exp_dir / best_name / "training" / "model.zip"
        reward_path = exp_dir / best_name / "training" / "reward_v1.py"

        if not model_path.exists():
            print(f"  seed_{seed}: best={best_name} but model MISSING, SKIP")
            continue

        dev_score = best_r["mean_eval_reward"]
        print(f"  seed_{seed}: best={best_name} (dev={dev_score:.2f})...", end=" ", flush=True)

        try:
            model = PPO.load(str(model_path))
            reward_fn = load_reward_fn(str(reward_path)) if reward_path.exists() else None
            ep_results = evaluate_model(model, ENV_ID, seeds, reward_fn)
            rewards = [r["reward"] for r in ep_results]
            lengths = [r["length"] for r in ep_results]
            terminated = sum(1 for r in ep_results if r["terminated"])

            seed_result = {
                "seed": f"seed_{seed}",
                "best_sample": best_name,
                "dev_score": dev_score,
                "held_out_mean": mean(rewards),
                "held_out_std": stdev(rewards) if len(rewards) > 1 else 0,
                "held_out_min": min(rewards),
                "held_out_max": max(rewards),
                "held_out_mean_len": mean(lengths),
                "held_out_terminated": terminated,
                "held_out_truncated": HELD_OUT_EPS - terminated,
                "solved": mean(rewards) >= SOLVED_THRESHOLD,
                "episodes": ep_results,
            }
            all_results[f"seed_{seed}"] = seed_result

            status = "[SOLVED]" if seed_result["solved"] else "       "
            print(f"{status} held-out={mean(rewards):.2f}+-{stdev(rewards) if len(rewards) > 1 else 0:.2f}  "
                  f"term={terminated}/{HELD_OUT_EPS}  len={mean(lengths):.1f}")
            del model
        except Exception as e:
            print(f"ERROR: {e}")

    output_file = output_dir / f"held_out_{exp_name}.json"
    json.dump(all_results, open(output_file, "w"), indent=2, ensure_ascii=False)
    write_summary_md(all_results, exp_name, output_dir)
    return all_results


def write_summary_md(all_results, exp_name, output_dir):
    if not all_results:
        return
    lines = [f"# Held-Out Evaluation: {exp_name}", "",
             f"- episodes: {HELD_OUT_EPS}", f"- seeds: {HELD_OUT_SEED_START}..{HELD_OUT_SEED_START + HELD_OUT_EPS - 1}",
             f"- env: {ENV_ID} (original reward)", f"- threshold: {SOLVED_THRESHOLD}", "",
             "| seed | dev_score | held_out_mean | held_out_std | min | max | len | term | solved |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]

    solved_count = 0
    total = 0
    held_scores = []
    for seed_name in sorted(all_results.keys()):
        r = all_results[seed_name]
        total += 1
        if r["solved"]:
            solved_count += 1
        held_scores.append(r["held_out_mean"])
        lines.append(
            f"| {seed_name} | {r.get('dev_score', '?'):.2f} | {r['held_out_mean']:.2f} | "
            f"{r['held_out_std']:.2f} | {r['held_out_min']:.1f} | {r['held_out_max']:.1f} | "
            f"{r['held_out_mean_len']:.1f} | {r['held_out_terminated']}/{HELD_OUT_EPS} | "
            f"{'yes' if r['solved'] else 'no'} |"
        )
    if held_scores:
        lines.extend(["", "## Summary", "",
                      f"- solved: {solved_count}/{total}",
                      f"- held_out_mean: {mean(held_scores):.2f}",
                      f"- held_out_std: {stdev(held_scores) if len(held_scores) > 1 else 0:.2f}"])
    (output_dir / f"held_out_{exp_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="analysis/held_out_eval")
    ap.add_argument("--experiments", nargs="*", default=None,
                    help="Which experiments to run (default: all)")
    args = ap.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(HELD_OUT_SEED_START, HELD_OUT_SEED_START + HELD_OUT_EPS))

    print("=" * 60)
    print("Unified Held-Out Evaluation")
    print(f"Env: {ENV_ID} (original reward, no wrapper)")
    print(f"Episodes: {HELD_OUT_EPS}  |  Seeds: {HELD_OUT_SEED_START}..{HELD_OUT_SEED_START + HELD_OUT_EPS - 1}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    to_run = args.experiments if args.experiments else list(EXPERIMENTS.keys())

    for exp_name in to_run:
        exp_cfg = EXPERIMENTS[exp_name]
        print(f"\n{'=' * 60}")
        print(f"Experiment: {exp_name} ({exp_cfg['dir']})")
        print(f"{'=' * 60}")

        if exp_cfg["type"] == "iterative":
            eval_iterative_experiment(exp_name, exp_cfg, seeds, output_dir)
        elif exp_cfg["type"] == "independent":
            eval_independent_experiment(exp_name, exp_cfg, seeds, output_dir)

    print(f"\nDone. Results in: {output_dir}")


if __name__ == "__main__":
    main()
