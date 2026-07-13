import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_root = Path(cfg["experiment"]["run_root"]) / args.prefix
    timesteps = args.total_timesteps or int(float(cfg["training"]["total_timesteps"]))

    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        save_dir = run_root / f"seed_{seed}" / "training"
        cmd = [
            sys.executable,
            "-m",
            "training.train_sb3_wrapper",
            "--config",
            args.config,
            "--use-original-reward",
            "--run-name",
            f"{args.prefix}_seed_{seed}",
            "--seed",
            str(seed),
            "--total-timesteps",
            str(timesteps),
            "--eval-episodes",
            str(args.eval_episodes),
            "--save-dir",
            str(save_dir),
        ]
        print(f"[official baseline] seed={seed} save_dir={save_dir}", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
