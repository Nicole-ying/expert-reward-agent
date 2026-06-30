"""Multi-seed iterative experiment runner.

Usage:
  python -m pipeline.run_multi_seed_experiment --config configs/env001_deepseek_rag.yaml

Config keys (all under multi_seed:):
  enabled    — must be true for this runner
  start_seed — first seed number (default 0)
  num_seeds  — how many seeds (default 10)
"""

import argparse
from .common import load_config
from .run_iterative_experiment import run_iterative_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ms_cfg = cfg.get("multi_seed", {})
    if not ms_cfg.get("enabled", False):
        raise RuntimeError(
            "multi_seed.enabled is not true. "
            "Use python -m pipeline.run_iterative_experiment for single-seed runs."
        )

    start_seed = int(ms_cfg.get("start_seed", 0))
    num_seeds = int(ms_cfg.get("num_seeds", 10))
    mock = True if args.mock else None

    print("=" * 60)
    print("Multi-seed iterative experiment")
    print("=" * 60)
    print(f"config          : {args.config}")
    print(f"prefix          : {args.prefix or cfg['iteration']['experiment_prefix']}")
    print(f"rounds          : {args.rounds or cfg['iteration']['total_rounds']}")
    print(f"start_seed      : {start_seed}")
    print(f"num_seeds       : {num_seeds}")
    print(f"total_timesteps : {args.total_timesteps or cfg['training']['total_timesteps']}")
    print(f"mock            : {mock or False}")

    for s in range(start_seed, start_seed + num_seeds):
        print("\n" + "#" * 60)
        print(f" SEED {s}  ({s - start_seed + 1} / {num_seeds})")
        print("#" * 60)
        run_iterative_experiment(
            config_path=args.config,
            prefix=args.prefix,
            rounds=args.rounds,
            total_timesteps=args.total_timesteps,
            eval_episodes=args.eval_episodes,
            mock=mock,
            seed=s,
        )

    print("\n" + "=" * 60)
    print("All seeds done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
