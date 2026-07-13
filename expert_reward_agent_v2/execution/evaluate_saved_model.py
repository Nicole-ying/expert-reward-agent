from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from stable_baselines3 import PPO

from training.reward_wrapper import load_reward_function
from training.train_sb3_wrapper import evaluate_model_on_original_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reward", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seed-offset", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    reward_fn = load_reward_function(args.reward)
    model = PPO.load(args.model)
    result = evaluate_model_on_original_env(
        model,
        config["training"]["runner_env_id"],
        eval_episodes=args.episodes,
        seed=args.seed,
        eval_seed_offset=args.seed_offset,
        reward_fn=reward_fn,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
