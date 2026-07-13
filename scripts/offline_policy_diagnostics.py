"""Replay a trained policy and collect shadow-mode reward/behavior diagnostics.

This tool never trains or changes a policy. Its component statistics describe the
fixed policy's evaluation distribution, not the policy's historical training data.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
import sys
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.reward_wrapper import load_reward_function


NON_COMPONENT_KEYS = {"total_reward", "generated_reward", "original_env_reward"}


class ScalarStats:
    def __init__(self) -> None:
        self.values: list[float] = []

    def add(self, value: Any) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(number):
            self.values.append(number)

    def summary(self) -> dict[str, Any]:
        if not self.values:
            return {"count": 0}
        active = [value for value in self.values if abs(value) > 1e-12]
        return {
            "count": len(self.values),
            "mean": mean(self.values),
            "abs_mean": mean(abs(value) for value in self.values),
            "nonzero_rate": len(active) / len(self.values),
            "mean_when_active": mean(active) if active else 0.0,
            "abs_mean_when_active": mean(abs(value) for value in active) if active else 0.0,
            "min": min(self.values),
            "max": max(self.values),
        }


def parse_reward_output(reward_out: Any) -> tuple[float, dict[str, float]]:
    if isinstance(reward_out, (tuple, list)) and len(reward_out) == 2:
        generated_reward, raw_terms = reward_out
    else:
        generated_reward, raw_terms = reward_out, {}
    generated_reward = float(generated_reward)
    terms: dict[str, float] = {}
    if isinstance(raw_terms, dict):
        # Future nested output is accepted while old flat component dictionaries
        # remain fully compatible.
        candidates = raw_terms.get("reward_terms", raw_terms)
        if isinstance(candidates, dict):
            for name, value in candidates.items():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    terms[str(name)] = number
        diagnostics = raw_terms.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            for name, value in diagnostics.items():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    terms[f"diagnostic.{name}"] = number
    return generated_reward, terms


def infer_additive_terms(
    step_records: list[dict[str, float]],
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Infer which legacy flat terms add up to generated_reward.

    The inference is evidence for reviewing legacy output, not a replacement for
    an explicit reward_terms/diagnostics schema in future generated rewards.
    """
    names = sorted(
        {
            name
            for row in step_records
            for name in row
            if name not in NON_COMPONENT_KEYS and not name.startswith("diagnostic.")
        }
    )
    if not step_records or not names or len(names) > 12:
        return {
            "status": "unavailable",
            "reward_terms": [],
            "diagnostics": names,
            "mean_abs_residual": None,
        }

    best_subset: tuple[str, ...] = ()
    best_residual = float("inf")
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            residual = mean(
                abs(row["generated_reward"] - sum(row.get(name, 0.0) for name in subset))
                for row in step_records
            )
            if residual < best_residual - 1e-15:
                best_residual = residual
                best_subset = subset
    scale = max(1.0, mean(abs(row["generated_reward"]) for row in step_records))
    status = "exact" if best_residual <= tolerance * scale else "uncertain"
    return {
        "status": status,
        "reward_terms": list(best_subset),
        "diagnostics": [name for name in names if name not in best_subset],
        "mean_abs_residual": best_residual,
        "relative_residual": best_residual / scale,
    }


def summarize_numbers(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    std = pstdev(values)
    return {
        "count": len(values),
        "mean": mean(values),
        "std": std,
        "ci95_half_width": 1.96 * std / math.sqrt(len(values)),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


class LanderDiagnostics:
    """Observation-only LunarLander facts; no crash/success ground-truth claims."""

    def __init__(self) -> None:
        self.distances: list[float] = []
        self.speeds: list[float] = []
        self.abs_angles: list[float] = []
        self.left_contacts: list[float] = []
        self.right_contacts: list[float] = []
        self.actions: list[int] = []

    def observe(self, next_obs: Any, action: Any) -> None:
        obs = np.asarray(next_obs, dtype=float).reshape(-1)
        if obs.size < 8:
            return
        self.distances.append(float(math.hypot(obs[0], obs[1])))
        self.speeds.append(float(math.hypot(obs[2], obs[3])))
        self.abs_angles.append(float(abs(obs[4])))
        self.left_contacts.append(float(obs[6] > 0.5))
        self.right_contacts.append(float(obs[7] > 0.5))
        self.actions.append(int(np.asarray(action).item()))

    def summary(self, terminated: bool, truncated: bool) -> dict[str, Any]:
        action_count = max(1, len(self.actions))
        action_rates = {
            str(action): self.actions.count(action) / action_count for action in range(4)
        }
        result = {
            "source": "derived_from_declared_observation_and_action",
            "outcome": (
                "timeout" if truncated else "terminated_unclassified" if terminated else "incomplete"
            ),
            "final_distance": self.distances[-1] if self.distances else None,
            "final_speed": self.speeds[-1] if self.speeds else None,
            "final_abs_angle": self.abs_angles[-1] if self.abs_angles else None,
            "left_contact_rate": mean(self.left_contacts) if self.left_contacts else None,
            "right_contact_rate": mean(self.right_contacts) if self.right_contacts else None,
            "both_contact_rate": (
                mean(left * right for left, right in zip(self.left_contacts, self.right_contacts))
                if self.left_contacts
                else None
            ),
            "action_rates": action_rates,
            "termination_reason": "unknown",
        }
        result.update({f"action_rate_{key}": value for key, value in action_rates.items()})
        return result


class BipedalDiagnostics:
    """Observation-only BipedalWalker facts without inventing terminal reasons."""

    def __init__(self) -> None:
        self.hull_angles: list[float] = []
        self.horizontal_velocities: list[float] = []
        self.vertical_velocities: list[float] = []
        self.left_contacts: list[float] = []
        self.right_contacts: list[float] = []
        self.action_abs: list[float] = []
        self.action_delta: list[float] = []
        self.previous_action: np.ndarray | None = None

    def observe(self, next_obs: Any, action: Any) -> None:
        obs = np.asarray(next_obs, dtype=float).reshape(-1)
        current_action = np.asarray(action, dtype=float).reshape(-1)
        if obs.size >= 14:
            self.hull_angles.append(float(obs[0]))
            self.horizontal_velocities.append(float(obs[2]))
            self.vertical_velocities.append(float(obs[3]))
            self.left_contacts.append(float(obs[12] > 0.5))
            self.right_contacts.append(float(obs[13] > 0.5))
        if current_action.size:
            self.action_abs.append(float(np.mean(np.abs(current_action))))
            if self.previous_action is not None and self.previous_action.shape == current_action.shape:
                self.action_delta.append(float(np.mean(np.abs(current_action - self.previous_action))))
            self.previous_action = current_action.copy()

    def summary(self, terminated: bool, truncated: bool) -> dict[str, Any]:
        return {
            "source": "derived_from_declared_observation_and_action",
            "outcome": (
                "timeout" if truncated else "terminated_unclassified" if terminated else "incomplete"
            ),
            "final_hull_angle": self.hull_angles[-1] if self.hull_angles else None,
            "mean_abs_hull_angle": (
                mean(abs(value) for value in self.hull_angles) if self.hull_angles else None
            ),
            "mean_horizontal_velocity": (
                mean(self.horizontal_velocities) if self.horizontal_velocities else None
            ),
            "final_horizontal_velocity": (
                self.horizontal_velocities[-1] if self.horizontal_velocities else None
            ),
            "mean_abs_vertical_velocity": (
                mean(abs(value) for value in self.vertical_velocities) if self.vertical_velocities else None
            ),
            "left_contact_rate": mean(self.left_contacts) if self.left_contacts else None,
            "right_contact_rate": mean(self.right_contacts) if self.right_contacts else None,
            "both_contact_rate": (
                mean(left * right for left, right in zip(self.left_contacts, self.right_contacts))
                if self.left_contacts
                else None
            ),
            "action_abs_mean": mean(self.action_abs) if self.action_abs else None,
            "action_delta_mean": mean(self.action_delta) if self.action_delta else None,
            "termination_reason": "unknown",
        }


def make_adapter(env_id: str) -> Any:
    if "LunarLander" in env_id:
        return LanderDiagnostics()
    if "BipedalWalker" in env_id:
        return BipedalDiagnostics()
    return None


def safe_reward_call(
    reward_fn: Any,
    obs: Any,
    action: Any,
    next_obs: Any,
    original_reward: float,
    terminated: bool,
    truncated: bool,
) -> tuple[float, dict[str, float], str | None]:
    safe_info = {
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done": bool(terminated or truncated),
    }
    try:
        reward_out = reward_fn(
            obs,
            action,
            next_obs,
            original_reward,
            safe_info,
            training_progress=1.0,
        )
        generated_reward, terms = parse_reward_output(reward_out)
        if not math.isfinite(generated_reward):
            raise ValueError("non-finite generated reward")
        return generated_reward, terms, None
    except Exception as exc:  # Keep a replay failure visible instead of aborting all episodes.
        return 0.0, {}, f"{type(exc).__name__}: {exc}"


def replay(
    model: PPO,
    reward_fn: Any,
    env_id: str,
    episodes: int,
    seed_base: int,
) -> dict[str, Any]:
    env = gym.make(env_id)
    episode_rows: list[dict[str, Any]] = []
    all_steps: list[dict[str, float]] = []
    component_stats: dict[str, ScalarStats] = {}
    component_episode_sums: dict[str, list[float]] = {}
    errors: list[dict[str, Any]] = []

    for episode_id in range(episodes):
        seed = seed_base + episode_id
        obs, _ = env.reset(seed=seed)
        adapter = make_adapter(env_id)
        terminated = truncated = False
        length = 0
        original_return = 0.0
        generated_return = 0.0
        episode_component_sums: dict[str, float] = {}

        while not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=True)
            next_obs, original_reward, terminated, truncated, _info = env.step(action)
            generated_reward, terms, error = safe_reward_call(
                reward_fn,
                obs,
                action,
                next_obs,
                float(original_reward),
                bool(terminated),
                bool(truncated),
            )
            if error:
                errors.append({"episode": episode_id, "step": length, "error": error})

            step_record = {"generated_reward": generated_reward}
            step_record.update(terms)
            all_steps.append(step_record)
            component_stats.setdefault("generated_reward", ScalarStats()).add(generated_reward)
            for name, value in terms.items():
                if name in NON_COMPONENT_KEYS:
                    continue
                component_stats.setdefault(name, ScalarStats()).add(value)
                episode_component_sums[name] = episode_component_sums.get(name, 0.0) + value

            original_return += float(original_reward)
            generated_return += generated_reward
            length += 1
            if adapter is not None:
                adapter.observe(next_obs, action)
            obs = next_obs

        for name in component_stats:
            if name == "generated_reward":
                continue
            component_episode_sums.setdefault(name, []).append(episode_component_sums.get(name, 0.0))

        episode_rows.append(
            {
                "episode": episode_id,
                "seed": seed,
                "original_return": original_return,
                "generated_return": generated_return,
                "length": length,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "component_sums": episode_component_sums,
                "behavior": adapter.summary(bool(terminated), bool(truncated)) if adapter else {},
            }
        )

    env.close()
    additive_inference = infer_additive_terms(all_steps)
    return {
        "metadata": {
            "distribution": "deterministic_policy_evaluation",
            "not_training_distribution": True,
            "env_id": env_id,
            "episodes": episodes,
            "seed_base": seed_base,
        },
        "evaluation": {
            "original_return": summarize_numbers([row["original_return"] for row in episode_rows]),
            "generated_return": summarize_numbers([row["generated_return"] for row in episode_rows]),
            "episode_length": summarize_numbers([float(row["length"]) for row in episode_rows]),
            "termination_rate": mean(float(row["terminated"]) for row in episode_rows),
            "truncation_rate": mean(float(row["truncated"]) for row in episode_rows),
        },
        "additive_term_inference": additive_inference,
        "step_component_stats": {
            name: stats.summary() for name, stats in sorted(component_stats.items())
        },
        "episode_component_sum_stats": {
            name: summarize_numbers(values) for name, values in sorted(component_episode_sums.items())
        },
        "episodes": episode_rows,
        "reward_errors": errors,
    }


def write_episode_csv(path: Path, report: dict[str, Any]) -> None:
    component_names = sorted(
        {name for row in report["episodes"] for name in row["component_sums"]}
    )
    behavior_names = sorted(
        {
            name
            for row in report["episodes"]
            for name, value in row["behavior"].items()
            if not isinstance(value, dict)
        }
    )
    fields = [
        "episode",
        "seed",
        "original_return",
        "generated_return",
        "length",
        "terminated",
        "truncated",
    ] + [f"component.{name}" for name in component_names] + [
        f"behavior.{name}" for name in behavior_names
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["episodes"]:
            flat = {name: row.get(name) for name in fields if name in row}
            for name in component_names:
                flat[f"component.{name}"] = row["component_sums"].get(name, 0.0)
            for name in behavior_names:
                flat[f"behavior.{name}"] = row["behavior"].get(name)
            writer.writerow(flat)


def fmt(value: Any) -> str:
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    meta = report["metadata"]
    evaluation = report["evaluation"]
    inference = report["additive_term_inference"]
    lines = [
        "# Offline Policy Diagnostics",
        "",
        "> Shadow-mode replay only. These statistics describe deterministic evaluation trajectories, not historical training trajectories.",
        "",
        "## Replay",
        "",
        f"- env_id: {meta['env_id']}",
        f"- episodes: {meta['episodes']}",
        f"- seed_base: {meta['seed_base']}",
        f"- reward_errors: {len(report['reward_errors'])}",
        "",
        "## Evaluation Summary",
        "",
        "| metric | mean | std | median | min | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("original_return", "generated_return", "episode_length"):
        row = evaluation[name]
        lines.append(
            f"| {name} | {fmt(row['mean'])} | {fmt(row['std'])} | {fmt(row['median'])} | {fmt(row['min'])} | {fmt(row['max'])} |"
        )
    lines.extend(
        [
            "",
            f"- termination_rate: {fmt(evaluation['termination_rate'])}",
            f"- truncation_rate: {fmt(evaluation['truncation_rate'])}",
            "",
            "## Additive-Term Inference",
            "",
            f"- status: {inference['status']}",
            f"- reward_terms: {', '.join(inference['reward_terms']) or 'none'}",
            f"- diagnostics/modulators: {', '.join(inference['diagnostics']) or 'none'}",
            f"- mean_abs_residual: {fmt(inference['mean_abs_residual'])}",
            "",
            "## Step Component Statistics",
            "",
            "| component | mean | abs_mean | active_rate | mean_when_active | abs_mean_when_active | min | max | count |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in report["step_component_stats"].items():
        lines.append(
            f"| {name} | {fmt(row.get('mean'))} | {fmt(row.get('abs_mean'))} | {fmt(row.get('nonzero_rate'))} | "
            f"{fmt(row.get('mean_when_active'))} | {fmt(row.get('abs_mean_when_active'))} | "
            f"{fmt(row.get('min'))} | {fmt(row.get('max'))} | {row.get('count', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Per-Episode Component Sums",
            "",
            "| component | mean | std | median | min | max | episodes |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in report["episode_component_sum_stats"].items():
        lines.append(
            f"| {name} | {fmt(row.get('mean'))} | {fmt(row.get('std'))} | {fmt(row.get('median'))} | "
            f"{fmt(row.get('min'))} | {fmt(row.get('max'))} | {row.get('count', 0)} |"
        )
    lines.extend(["", "## Episodes", "", "| episode | seed | original_return | generated_return | length | terminated | truncated | outcome | final_distance | final_speed | both_contact_rate |", "|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|"])
    for row in report["episodes"]:
        behavior = row["behavior"]
        lines.append(
            f"| {row['episode']} | {row['seed']} | {fmt(row['original_return'])} | {fmt(row['generated_return'])} | "
            f"{row['length']} | {row['terminated']} | {row['truncated']} | {behavior.get('outcome', 'n/a')} | "
            f"{fmt(behavior.get('final_distance'))} | {fmt(behavior.get('final_speed'))} | {fmt(behavior.get('both_contact_rate'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_inputs(run_dir: Path) -> tuple[Path, Path, str, int]:
    config_path = run_dir / "train_config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_path = Path(config.get("model_path", run_dir / "model.zip"))
    reward_path = Path(config["reward_path"])
    env_id = str(config["runner_env_id"])
    ppo_seed = int(config.get("ppo_args", {}).get("seed", 0))
    return model_path, reward_path, env_id, ppo_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    model_path, reward_path, env_id, ppo_seed = resolve_inputs(args.run_dir)
    seed_base = args.seed_base if args.seed_base is not None else ppo_seed + 10000
    output_dir = args.output_dir or args.run_dir / "offline_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(str(model_path))
    reward_fn = load_reward_function(str(reward_path))
    report = replay(model, reward_fn, env_id, args.episodes, seed_base)
    report["metadata"].update(
        {
            "run_dir": str(args.run_dir),
            "model_path": str(model_path),
            "reward_path": str(reward_path),
        }
    )

    (output_dir / "offline_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_episode_csv(output_dir / "episodes.csv", report)
    write_markdown(output_dir / "offline_diagnostics.md", report)
    print(f"Wrote offline diagnostics to: {output_dir}")
    print(f"Original return mean: {report['evaluation']['original_return']['mean']:.6f}")
    print(f"Additive-term inference: {report['additive_term_inference']}")


if __name__ == "__main__":
    main()
