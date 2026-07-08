from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]


def parse_memory(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    if not path.exists():
        return rows
    headers = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == "iter":
            headers = cells
            continue
        if not headers or not cells or not cells[0].isdigit() or len(cells) != len(headers):
            continue
        rows[int(cells[0])] = dict(zip(headers, cells))
    return rows


def parse_component_means(feedback_path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    if not feedback_path.exists():
        return values
    in_table = False
    for line in feedback_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| component |"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and not line.startswith("|"):
            break
        if not in_table:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] in {"total_reward", "generated_reward", "original_env_reward"}:
            continue
        try:
            values[cells[0]] = float(cells[1])
        except ValueError:
            pass
    return values


def reward_version(iter_dir: Path) -> str:
    files = sorted((iter_dir / "generation").glob("reward_v*.py"))
    return files[0].stem if files else ""


def collect(manifest_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    run_root = ROOT / manifest["run_root"]
    threshold = float(manifest["solved_threshold"])
    records = []
    components = []

    for seed in manifest["seeds"]:
        seed_dir = run_root / f"seed_{seed}"
        memory = parse_memory(seed_dir / "memory" / "reward_memory.md")
        running_best = float("-inf")
        for iteration in range(1, int(manifest["max_iterations"]) + 1):
            iter_dir = seed_dir / f"iter_{iteration:02d}"
            eval_path = iter_dir / "training" / "eval_result.json"
            if not eval_path.exists():
                continue
            result = json.loads(eval_path.read_text(encoding="utf-8"))
            score = float(result["mean_eval_reward"])
            running_best = max(running_best, score)
            memory_row = memory.get(iteration, {})
            rewards = [float(value) for value in result.get("episode_rewards", [])]
            records.append(
                {
                    "dataset_id": manifest["dataset_id"],
                    "environment": manifest["environment"],
                    "method": manifest["method"],
                    "seed": int(seed),
                    "iteration": iteration,
                    "score": score,
                    "score_std": pd.Series(rewards).std(ddof=1) if len(rewards) > 1 else float("nan"),
                    "score_min": float(result.get("min_eval_reward", min(rewards) if rewards else score)),
                    "score_max": float(result.get("max_eval_reward", max(rewards) if rewards else score)),
                    "episode_length": float(result["mean_episode_length"]),
                    "eval_episodes": int(result.get("eval_episodes", len(rewards))),
                    "best_so_far": running_best,
                    "solved": score >= threshold,
                    "solved_so_far": running_best >= threshold,
                    "skeleton": memory_row.get("skeleton", ""),
                    "decision": memory_row.get("action", ""),
                    "restart_scheduled": "fresh_restart" in memory_row.get("action", ""),
                    "reward_version": reward_version(iter_dir),
                    "run_dir": str(iter_dir.relative_to(ROOT)).replace("\\", "/"),
                }
            )
            feedback_path = iter_dir / "training" / "training_feedback.md"
            for name, mean in parse_component_means(feedback_path).items():
                components.append(
                    {
                        "seed": int(seed),
                        "iteration": iteration,
                        "component": name,
                        "mean": mean,
                    }
                )

    frame = pd.DataFrame(records).sort_values(["seed", "iteration"]).reset_index(drop=True)
    component_frame = pd.DataFrame(components)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "iteration_results.csv", index=False)
    component_frame.to_csv(output_dir / "component_means.csv", index=False)

    seed_rows = []
    for seed, group in frame.groupby("seed"):
        solved = group[group["score"] >= threshold]
        best_idx = group["score"].idxmax()
        seed_rows.append(
            {
                "seed": int(seed),
                "iterations_available": len(group),
                "initial_score": float(group.iloc[0]["score"]),
                "best_score": float(group.loc[best_idx, "score"]),
                "best_iteration": int(group.loc[best_idx, "iteration"]),
                "first_solved_iteration": int(solved.iloc[0]["iteration"]) if not solved.empty else "",
                "solved": not solved.empty,
                "eval_episode_counts": ",".join(map(str, sorted(group["eval_episodes"].unique()))),
            }
        )
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(output_dir / "seed_summary.csv", index=False)

    gaps = []
    for seed, group in frame.groupby("seed"):
        observed = set(group["iteration"].astype(int))
        expected = set(range(int(group["iteration"].min()), int(group["iteration"].max()) + 1))
        missing = sorted(expected - observed)
        if missing:
            gaps.append(f"- seed {seed}: missing iterations {missing}")
    report = [
        "# V6 Exploratory Data Quality Report",
        "",
        f"- seeds: {manifest['seeds']}",
        f"- observed iterations: {len(frame)}",
        f"- solved seeds: {int(seeds['solved'].sum())}/{len(seeds)}",
        f"- evaluation episode counts found: {sorted(frame['eval_episodes'].unique().tolist())}",
        "- best_so_far is recomputed from observed scores; stored historical best values are not trusted.",
        "- status: exploratory only; regenerate all figures from frozen final experiments for submission.",
        "",
        "## Missing Iterations",
        *(gaps or ["- none"]),
    ]
    (output_dir / "data_quality_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return frame, component_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "analysis/paper/experiment_manifest.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/paper/v6_exploratory")
    args = parser.parse_args()
    frame, _ = collect(args.manifest, args.output_dir)
    print(f"collected {len(frame)} iteration records -> {args.output_dir}")


if __name__ == "__main__":
    main()

