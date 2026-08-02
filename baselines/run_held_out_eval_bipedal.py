#!/usr/bin/env python3
"""Held-out evaluation for BipedalWalker-v3 (env_002).
Same 100 seeds (50000-50099) as LunarLander experiments.
"""
import json, sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

ENV_ID = "BipedalWalker-v3"
HELD_OUT_EPS = 100
HELD_OUT_SEED_START = 50000
SOLVED_THRESHOLD = 300.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "runs" / "env_002" / "paper_bipedal_main_v1"


def load_reward_fn(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("reward", str(Path(path)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_reward


def main():
    output_dir = PROJECT_ROOT / "analysis" / "held_out_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(HELD_OUT_SEED_START, HELD_OUT_SEED_START + HELD_OUT_EPS))

    print("=" * 60)
    print("BipedalWalker-v3 Held-Out Evaluation")
    print(f"Episodes: {HELD_OUT_EPS}  |  Seeds: {HELD_OUT_SEED_START}..{HELD_OUT_SEED_START + HELD_OUT_EPS - 1}")
    print(f"Threshold: {SOLVED_THRESHOLD}")
    print("=" * 60)

    all_results = {}

    for seed_dir in sorted(EXP_DIR.glob("seed_*")):
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
            print(f"  {seed_name}: model MISSING, SKIP")
            continue

        dev_str = f"{dev_score:.2f}" if dev_score is not None else "?"
        print(f"  {seed_name}: loading model (dev={dev_str})...", end=" ", flush=True)

        try:
            model = PPO.load(str(model_path))

            episode_rewards, episode_lengths, episode_terminated = [], [], []
            for ep_seed in seeds:
                env = gym.make(ENV_ID)
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
                episode_rewards.append(ep_r)
                episode_lengths.append(ep_l)
                episode_terminated.append(was_terminated)

            rewards = episode_rewards
            lengths = episode_lengths
            terminated = sum(1 for t in episode_terminated if t)

            seed_result = {
                "seed": seed_name,
                "dev_score": dev_score,
                "held_out_mean": mean(rewards),
                "held_out_std": stdev(rewards) if len(rewards) > 1 else 0,
                "held_out_min": min(rewards),
                "held_out_max": max(rewards),
                "held_out_mean_len": mean(lengths),
                "held_out_terminated": terminated,
                "held_out_truncated": HELD_OUT_EPS - terminated,
                "solved": mean(rewards) >= SOLVED_THRESHOLD,
                "episodes": [{"seed": s, "reward": r, "length": l, "terminated": t}
                             for s, r, l, t in zip(seeds, rewards, lengths, episode_terminated)],
            }
            all_results[seed_name] = seed_result

            status = "[SOLVED]" if seed_result["solved"] else "       "
            print(f"{status} held-out={mean(rewards):.2f}+-{stdev(rewards) if len(rewards) > 1 else 0:.2f}  "
                  f"term={terminated}/{HELD_OUT_EPS}  len={mean(lengths):.1f}")
            del model
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Write results
    json.dump(all_results, open(output_dir / "held_out_BipedalWalker.json", "w"),
              indent=2, ensure_ascii=False)

    # Write summary MD
    lines = ["# Held-Out Evaluation: BipedalWalker (CREATE)", "",
             f"- env: {ENV_ID} (original reward)", f"- episodes: {HELD_OUT_EPS}",
             f"- seeds: {HELD_OUT_SEED_START}..{HELD_OUT_SEED_START + HELD_OUT_EPS - 1}",
             f"- threshold: {SOLVED_THRESHOLD}", "",
             "| seed | dev_score | held_out_mean | held_out_std | min | max | len | term | solved |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]

    solved_count = 0
    held_scores = []
    for seed_name in sorted(all_results.keys()):
        r = all_results[seed_name]
        if r["solved"]:
            solved_count += 1
        held_scores.append(r["held_out_mean"])
        lines.append(
            f"| {seed_name} | {r['dev_score']:.2f} | {r['held_out_mean']:.2f} | "
            f"{r['held_out_std']:.2f} | {r['held_out_min']:.1f} | {r['held_out_max']:.1f} | "
            f"{r['held_out_mean_len']:.1f} | {r['held_out_terminated']}/{HELD_OUT_EPS} | "
            f"{'yes' if r['solved'] else 'no'} |"
        )

    if held_scores:
        lines.extend(["", "## Summary", "",
                      f"- solved: {solved_count}/{len(all_results)}",
                      f"- held_out_mean: {mean(held_scores):.2f}",
                      f"- held_out_std: {stdev(held_scores) if len(held_scores) > 1 else 0:.2f}"])

    (output_dir / "held_out_BipedalWalker.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nDone. Results in: {output_dir}")
    print(f"  held_out_BipedalWalker.json")
    print(f"  held_out_BipedalWalker.md")


if __name__ == "__main__":
    main()
