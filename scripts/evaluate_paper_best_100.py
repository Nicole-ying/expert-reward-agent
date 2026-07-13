import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from training.reward_wrapper import load_reward_function
from training.train_sb3_wrapper import evaluate_model_on_original_env


def resolve_existing(path_value, repo_root):
    path = Path(path_value)
    if path.exists():
        return path
    candidate = repo_root / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(path_value)


def write_seed_outputs(output_dir, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rewards = [float(value) for value in result.get("episode_rewards", [])]
    lengths = result.get("episode_lengths", [])
    seeds = result.get("eval_seeds", [])
    with (output_dir / "evaluation_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "eval_seed", "original_reward", "episode_length"])
        for index, reward in enumerate(rewards):
            writer.writerow([
                index,
                seeds[index] if index < len(seeds) else "",
                reward,
                lengths[index] if index < len(lengths) else "",
            ])
    std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(len(rewards)) if rewards else 0.0
    lines = [
        "# Independent 100-Episode Evaluation",
        "",
        f"- episodes: {len(rewards)}",
        f"- eval_seed_start: {seeds[0] if seeds else 'N/A'}",
        f"- eval_seed_end: {seeds[-1] if seeds else 'N/A'}",
        f"- mean_reward: {statistics.mean(rewards):.6f}",
        f"- sample_std: {std:.6f}",
        f"- ci95_half_width: {ci95:.6f}",
        f"- median_reward: {statistics.median(rewards):.6f}",
        f"- min_reward: {min(rewards):.6f}",
        f"- max_reward: {max(rewards):.6f}",
    ]
    (output_dir / "eval_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "mean": statistics.mean(rewards),
        "std": std,
        "ci95": ci95,
        "median": statistics.median(rewards),
        "min": min(rewards),
        "max": max(rewards),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    parser.add_argument("--experiment-root", default="runs/env_001/lander_v2_20260704")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seed-offset", type=int, default=30000)
    args = parser.parse_args()

    repo_root = Path.cwd()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    env_id = cfg["training"]["runner_env_id"]
    root = Path(args.experiment_root)
    aggregate = []

    for seed_dir in sorted(root.glob("seed_*")):
        best_summary_path = seed_dir / "best" / "best_training_summary.json"
        best_summary = json.loads(best_summary_path.read_text(encoding="utf-8"))
        model_path = resolve_existing(best_summary["model_path"], repo_root)
        reward_path = resolve_existing(best_summary["reward_path"], repo_root)
        model = PPO.load(str(model_path))
        reward_fn = load_reward_function(str(reward_path))
        observation_normalizer = None
        normalization = best_summary.get("normalization", {})
        normalize_env = None
        if normalization.get("normalize_obs"):
            stats_path = resolve_existing(normalization["vecnormalize_path"], repo_root)
            normalize_env = DummyVecEnv([lambda: gym.make(env_id)])
            observation_normalizer = VecNormalize.load(str(stats_path), normalize_env)
            observation_normalizer.training = False
            observation_normalizer.norm_reward = False
        result = evaluate_model_on_original_env(
            model,
            env_id,
            eval_episodes=args.episodes,
            seed=0,
            eval_seed_offset=args.eval_seed_offset,
            reward_fn=reward_fn,
            observation_normalizer=observation_normalizer,
        )
        if observation_normalizer is not None:
            observation_normalizer.close()
        stats = write_seed_outputs(seed_dir / "best" / "evaluation_100", result)
        aggregate.append({"seed": int(seed_dir.name.split("_")[-1]), **stats})
        print(f"{seed_dir.name}: {stats['mean']:.3f} +/- {stats['ci95']:.3f}", flush=True)

    aggregate.sort(key=lambda item: item["seed"])
    (root / "best_100_evaluation_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (root / "best_100_evaluation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "mean", "std", "ci95", "median", "min", "max"])
        writer.writeheader()
        writer.writerows(aggregate)


if __name__ == "__main__":
    main()
