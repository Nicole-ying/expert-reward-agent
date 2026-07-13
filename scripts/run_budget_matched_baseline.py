"""Budget-Matched Independent Search baseline.

Generates N independent reward functions (no iteration, no reflection),
trains each with PPO, and reports the best score.

This matches the compute budget of paper_v4's iterative search:
  10 iterations x 1 training = 10 independent generations x 1 training.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None


def _load_config(path):
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        cfg = yaml.safe_load(text) or {}
        parent = cfg.pop("extends", None)
        if parent:
            parent_path = Path(parent)
            if not parent_path.is_absolute():
                parent_path = config_path.parent / parent_path
            base = _load_config(parent_path)

            def _deep_merge(b, o):
                r = dict(b)
                for k, v in o.items():
                    if isinstance(v, dict) and isinstance(r.get(k), dict):
                        r[k] = _deep_merge(r[k], v)
                    else:
                        r[k] = v
                return r
            cfg = _deep_merge(base, cfg)
        return cfg
    raise RuntimeError("PyYAML is required")


def run_cmd(cmd, timeout=None):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_paper_v4.yaml")
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--total-timesteps", type=int, default=1000000)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--out-dir", default="budget_matched_independent_v1")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    run_root = Path(cfg["experiment"]["run_root"])
    out_dir = run_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = []
    for i in range(args.num_samples):
        run_name = f"{args.out_dir}/sample_{i:02d}"
        gen_dir = run_root / run_name / "generation"
        train_dir = run_root / run_name / "training"

        print(f"\n{'='*60}")
        print(f"Sample {i}/{args.num_samples-1}")
        print(f"{'='*60}", flush=True)

        # Step 1: generate initial reward
        run_cmd([
            "python", "-m", "pipeline.run_direct_generation_pipeline",
            "--config", args.config,
            "--run-name", f"{run_name}/generation",
            "--seed", str(i),
        ])

        # Step 2: train
        reward_py = gen_dir / "reward_v1.py"
        if not reward_py.exists():
            print(f"WARNING: {reward_py} not found, skipping sample {i}")
            scores.append(None)
            continue

        run_cmd([
            "python", "-m", "training.train_sb3_wrapper",
            "--config", args.config,
            "--reward", str(reward_py),
            "--run-name", f"{run_name}/training",
            "--save-dir", str(train_dir),
            "--total-timesteps", str(args.total_timesteps),
            "--eval-episodes", str(args.eval_episodes),
            "--seed", str(i),
        ])

        # Step 3: read score
        summary_path = train_dir / "training_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            score = float(summary.get("external_eval", {}).get("mean_eval_reward", 0.0))
            scores.append(score)
            print(f"Sample {i}: score={score:.3f}", flush=True)
        else:
            print(f"WARNING: {summary_path} not found for sample {i}")
            scores.append(None)

    # Write summary
    valid = [s for s in scores if s is not None]
    summary_text = f"""# Budget-Matched Independent Search

- num_samples: {args.num_samples}
- config: {args.config}

## Results

| sample | score | solved (>=200) |
|---|---:|---:|
"""
    for i, s in enumerate(scores):
        if s is not None:
            summary_text += f"| {i} | {s:.3f} | {'yes' if s >= 200 else 'no'} |\n"
        else:
            summary_text += f"| {i} | N/A | N/A |\n"

    if valid:
        best = max(valid)
        mean = sum(valid) / len(valid)
        std = (sum((s-mean)**2 for s in valid)/len(valid))**0.5
        solved = sum(1 for s in valid if s >= 200)
        summary_text += f"""
## Summary

| metric | value |
|---|---:|
| best | {best:.3f} |
| mean | {mean:.3f} |
| std | {std:.3f} |
| solved | {solved}/{len(valid)} |
"""
        print(f"\nBest: {best:.3f}, Mean: {mean:.3f}, Solved: {solved}/{len(valid)}")

    (out_dir / "summary.md").write_text(summary_text, encoding="utf-8")
    print(f"\nResults saved to {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
