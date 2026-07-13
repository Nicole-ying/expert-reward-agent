import argparse
import json
import subprocess
import sys
from pathlib import Path

from pipeline.common import load_config


def run_checked(command):
    subprocess.run(command, check=True)


def generate_candidate(config, run_name, generation_seed, max_retries=3):
    base_command = [
        sys.executable,
        "-m",
        "pipeline.run_direct_generation_pipeline",
        "--config",
        config,
        "--run-name",
        run_name,
        "--seed",
        str(generation_seed),
    ]
    validation_error = None
    for attempt in range(max_retries):
        command = list(base_command)
        if validation_error:
            command.extend(["--validation-retry", validation_error])
        run_checked(command)
        cfg = load_config(config)
        run_dir = Path(cfg["experiment"]["run_root"]) / run_name
        validation_path = run_dir / "validations" / "reward_v1.validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("valid"):
            return run_dir / "reward_v1.py"
        validation_error = "; ".join(validation.get("errors", [])) or "invalid reward code"
        print(f"candidate validation retry {attempt + 1}/{max_retries}: {validation_error}", flush=True)
    raise RuntimeError(f"Candidate remained invalid after {max_retries} attempts: {run_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_root = Path(cfg["experiment"]["run_root"])
    timesteps = args.total_timesteps or int(float(cfg["training"]["total_timesteps"]))

    for search_seed in range(args.start_seed, args.start_seed + args.num_seeds):
        seed_records = []
        for candidate_index in range(1, args.candidates + 1):
            relative_root = f"{args.prefix}/seed_{search_seed}/candidate_{candidate_index:02d}"
            generation_name = f"{relative_root}/generation"
            reward_path = generate_candidate(
                args.config,
                generation_name,
                generation_seed=search_seed * 1000 + candidate_index,
            )
            training_dir = run_root / relative_root / "training"
            run_checked([
                sys.executable,
                "-m",
                "training.train_sb3_wrapper",
                "--config",
                args.config,
                "--reward",
                str(reward_path),
                "--run-name",
                f"{args.prefix}_s{search_seed}_c{candidate_index}",
                "--seed",
                str(search_seed),
                "--total-timesteps",
                str(timesteps),
                "--eval-episodes",
                str(args.eval_episodes),
                "--save-dir",
                str(training_dir),
            ])
            result = json.loads((training_dir / "eval_result.json").read_text(encoding="utf-8"))
            record = {
                "candidate": candidate_index,
                "score": result["mean_eval_reward"],
                "reward_path": str(reward_path),
                "training_dir": str(training_dir),
            }
            seed_records.append(record)
            summary_path = run_root / args.prefix / f"seed_{search_seed}" / "candidate_results.json"
            summary_path.write_text(json.dumps(seed_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                f"[independent] seed={search_seed} candidate={candidate_index} "
                f"score={record['score']:.3f}",
                flush=True,
            )

        best = max(seed_records, key=lambda item: item["score"])
        summary = {
            "search_seed": search_seed,
            "candidate_budget": args.candidates,
            "best_candidate": best["candidate"],
            "best_score": best["score"],
            "candidates": seed_records,
        }
        summary_path = run_root / args.prefix / f"seed_{search_seed}" / "experiment_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
