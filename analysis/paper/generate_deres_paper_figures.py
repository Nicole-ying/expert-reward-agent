from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.paper.paper_style import COLORS, apply_paper_style, clean_axis, save_figure


ROOT = Path(__file__).resolve().parents[2]

ENV_THRESHOLDS = {
    "env_001": 200.0,
    "env_002": 300.0,
    "env_005": 2000.0,
}

RUN_SPECS = [
    {
        "env": "env_001",
        "method": "DERES",
        "run_rel": "runs/env_001/paper_v4",
        "kind": "iterative",
        "role": "main",
    },
    {
        "env": "env_001",
        "method": "w/o HEP",
        "run_rel": "runs/env_001/ablation_unconstrained_v4",
        "kind": "iterative",
        "role": "ablation",
    },
    {
        "env": "env_001",
        "method": "Budget-Matched Independent",
        "run_rel": "runs/env_001/budget_matched_independent_v1",
        "kind": "independent",
        "role": "ablation",
    },
    {
        "env": "env_002",
        "method": "DERES",
        "run_rel": "runs/env_002/paper_bipedal_main_v1",
        "kind": "iterative",
        "role": "main",
    },
    {
        "env": "env_002",
        "method": "Chat + Env Card",
        "run_rel": "runs/env_002/test_chat_with_env_card",
        "kind": "iterative",
        "role": "env_understanding",
    },
    {
        "env": "env_002",
        "method": "No Expert Profile",
        "run_rel": "runs/env_002/ablation_no_expert_profile_v1",
        "kind": "iterative",
        "role": "env_understanding",
    },
    {
        "env": "env_002",
        "method": "DeepSeek Direct",
        "run_rel": "runs/env_002/ablation_direct_no_expert_v2",
        "kind": "direct_json",
        "role": "env_understanding",
    },
    {
        "env": "env_002",
        "method": "Direct No Env Card",
        "run_rel": "runs/env_002/ablation_direct_no_env_card_v1",
        "kind": "direct_json",
        "role": "env_understanding",
    },
    {
        "env": "env_002",
        "method": "Chat Direct",
        "run_rel": "runs/env_002/ablation_direct_no_expert_v3_chat",
        "kind": "direct_json",
        "role": "env_understanding",
    },
    {
        "env": "env_005",
        "method": "DERES",
        "run_rel": "runs/env_005/paper_ant_v7",
        "kind": "iterative",
        "role": "case",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_index(name: str, prefix: str) -> int | None:
    match = re.match(rf"{re.escape(prefix)}_(\d+)$", name)
    return int(match.group(1)) if match else None


def parse_memory(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    if not path.exists():
        return rows
    headers: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == "iter":
            headers = cells
            continue
        if not headers or not cells or len(cells) != len(headers):
            continue
        if not cells[0].isdigit():
            continue
        rows[int(cells[0])] = dict(zip(headers, cells))
    return rows


def load_eval_payload(training_dir: Path) -> dict[str, Any] | None:
    eval_path = training_dir / "eval_result.json"
    if eval_path.exists():
        return read_json(eval_path)
    summary_path = training_dir / "training_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        external = summary.get("external_eval")
        if isinstance(external, dict):
            return external
    return None


def reward_version(iter_dir: Path) -> str:
    files = sorted((iter_dir / "generation").glob("reward_v*.py"))
    return files[0].stem if files else ""


def eval_score(payload: dict[str, Any]) -> float:
    if "mean_eval_reward" in payload:
        return safe_float(payload["mean_eval_reward"])
    final_eval = payload.get("final_eval")
    if isinstance(final_eval, dict):
        return safe_float(final_eval.get("mean_reward", final_eval.get("mean_eval_reward")))
    return safe_float(payload.get("mean_reward"))


def eval_length(payload: dict[str, Any]) -> float:
    if "mean_episode_length" in payload:
        return safe_float(payload["mean_episode_length"])
    if "mean_ep_len" in payload:
        return safe_float(payload["mean_ep_len"])
    final_eval = payload.get("final_eval")
    if isinstance(final_eval, dict):
        return safe_float(final_eval.get("mean_ep_len", final_eval.get("mean_episode_length")))
    lengths = payload.get("episode_lengths") or payload.get("lengths") or []
    return float(np.mean(lengths)) if lengths else float("nan")


def eval_rewards(payload: dict[str, Any]) -> list[float]:
    rewards = payload.get("episode_rewards") or payload.get("rewards") or []
    return [safe_float(value) for value in rewards if value is not None]


def eval_components(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    raw = payload.get("final_policy_component_evaluation", {})
    if not raw and isinstance(payload.get("external_eval"), dict):
        raw = payload["external_eval"].get("final_policy_component_evaluation", {})
    components: dict[str, dict[str, float]] = {}
    if not isinstance(raw, dict):
        return components
    for name, metrics in raw.items():
        if name in {"total_reward", "generated_reward", "original_env_reward"}:
            continue
        if not isinstance(metrics, dict):
            continue
        components[name] = {
            "episode_sum_mean": safe_float(metrics.get("episode_sum_mean")),
            "episode_abs_sum_mean": safe_float(metrics.get("episode_abs_sum_mean")),
            "signed_contribution_share": safe_float(metrics.get("signed_contribution_share")),
            "magnitude_share": safe_float(metrics.get("magnitude_share")),
            "active_rate": safe_float(metrics.get("active_rate")),
        }
    return components


def collect_iterative(spec: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = ROOT / spec["run_rel"]
    records: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    if not run_dir.exists():
        return records, components

    threshold = ENV_THRESHOLDS[spec["env"]]
    for seed_dir in sorted(run_dir.glob("seed_*")):
        seed = extract_index(seed_dir.name, "seed")
        if seed is None:
            continue
        memory = parse_memory(seed_dir / "memory" / "reward_memory.md")
        running_best = float("-inf")
        for iter_dir in sorted(seed_dir.glob("iter_*")):
            iteration = extract_index(iter_dir.name, "iter")
            if iteration is None:
                continue
            payload = load_eval_payload(iter_dir / "training")
            mem = memory.get(iteration, {})
            if payload is None and not mem:
                continue
            score = eval_score(payload) if payload is not None else safe_float(mem.get("score"))
            if math.isnan(score):
                continue
            rewards = eval_rewards(payload) if payload is not None else []
            running_best = max(running_best, score)
            records.append(
                {
                    "env": spec["env"],
                    "method": spec["method"],
                    "role": spec["role"],
                    "run": spec["run_rel"],
                    "unit": f"seed_{seed}",
                    "seed": seed,
                    "sample": np.nan,
                    "iteration": iteration,
                    "score": score,
                    "score_std": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else float("nan"),
                    "score_min": safe_float(payload.get("min_eval_reward"), min(rewards) if rewards else score) if payload is not None else score,
                    "score_max": safe_float(payload.get("max_eval_reward"), max(rewards) if rewards else score) if payload is not None else score,
                    "episode_length": eval_length(payload) if payload is not None else safe_float(mem.get("len")),
                    "eval_episodes": int(payload.get("eval_episodes", payload.get("n_episodes", len(rewards)) or 0)) if payload is not None else 0,
                    "threshold": threshold,
                    "solved": score >= threshold,
                    "best_so_far": running_best,
                    "solved_so_far": running_best >= threshold,
                    "skeleton": mem.get("skeleton", ""),
                    "decision": mem.get("action", ""),
                    "reward_version": reward_version(iter_dir),
                    "path": str(iter_dir.relative_to(ROOT)).replace("\\", "/"),
                }
            )
            payload_components = eval_components(payload) if payload is not None else {}
            for component_name, metrics in payload_components.items():
                row = {
                    "env": spec["env"],
                    "method": spec["method"],
                    "run": spec["run_rel"],
                    "seed": seed,
                    "sample": np.nan,
                    "iteration": iteration,
                    "component": component_name,
                }
                row.update(metrics)
                components.append(row)
    return records, components


def collect_independent(spec: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = ROOT / spec["run_rel"]
    records: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    if not run_dir.exists():
        return records, components
    threshold = ENV_THRESHOLDS[spec["env"]]
    running_best = float("-inf")
    for sample_dir in sorted(run_dir.glob("sample_*")):
        sample = extract_index(sample_dir.name, "sample")
        if sample is None:
            continue
        payload = load_eval_payload(sample_dir / "training")
        if payload is None:
            continue
        score = eval_score(payload)
        if math.isnan(score):
            continue
        rewards = eval_rewards(payload)
        running_best = max(running_best, score)
        records.append(
            {
                "env": spec["env"],
                "method": spec["method"],
                "role": spec["role"],
                "run": spec["run_rel"],
                "unit": f"sample_{sample}",
                "seed": np.nan,
                "sample": sample,
                "iteration": sample + 1,
                "score": score,
                "score_std": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else float("nan"),
                "score_min": safe_float(payload.get("min_eval_reward"), min(rewards) if rewards else score),
                "score_max": safe_float(payload.get("max_eval_reward"), max(rewards) if rewards else score),
                "episode_length": eval_length(payload),
                "eval_episodes": int(payload.get("eval_episodes", payload.get("n_episodes", len(rewards)) or 0)),
                "threshold": threshold,
                "solved": score >= threshold,
                "best_so_far": running_best,
                "solved_so_far": running_best >= threshold,
                "skeleton": "",
                "decision": "independent_sample",
                "reward_version": reward_version(sample_dir),
                "path": str(sample_dir.relative_to(ROOT)).replace("\\", "/"),
            }
        )
        for component_name, metrics in eval_components(payload).items():
            row = {
                "env": spec["env"],
                "method": spec["method"],
                "run": spec["run_rel"],
                "seed": np.nan,
                "sample": sample,
                "iteration": sample + 1,
                "component": component_name,
            }
            row.update(metrics)
            components.append(row)
    return records, components


def collect_direct_json(spec: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = ROOT / spec["run_rel"]
    result_path = run_dir / "results.json"
    if not result_path.exists():
        return [], []
    payload = read_json(result_path)
    final_eval = payload.get("final_eval", {})
    score = eval_score(payload)
    rewards = eval_rewards(payload)
    threshold = ENV_THRESHOLDS[spec["env"]]
    return [
        {
            "env": spec["env"],
            "method": spec["method"],
            "role": spec["role"],
            "run": spec["run_rel"],
            "unit": "seed_0",
            "seed": safe_float(payload.get("seed", 0)),
            "sample": np.nan,
            "iteration": 1,
            "score": score,
            "score_std": safe_float(final_eval.get("std_reward"), float(np.std(rewards, ddof=1)) if len(rewards) > 1 else np.nan),
            "score_min": safe_float(final_eval.get("min"), min(rewards) if rewards else score),
            "score_max": safe_float(final_eval.get("max"), max(rewards) if rewards else score),
            "episode_length": eval_length(payload),
            "eval_episodes": int(final_eval.get("n_episodes", len(rewards) or 1)),
            "threshold": threshold,
            "solved": score >= threshold,
            "best_so_far": score,
            "solved_so_far": score >= threshold,
            "skeleton": "",
            "decision": "direct_generation",
            "reward_version": "reward_v1",
            "path": str(result_path.relative_to(ROOT)).replace("\\", "/"),
        }
    ], []


def collect_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    for spec in RUN_SPECS:
        if spec["kind"] == "iterative":
            run_records, run_components = collect_iterative(spec)
        elif spec["kind"] == "independent":
            run_records, run_components = collect_independent(spec)
        elif spec["kind"] == "direct_json":
            run_records, run_components = collect_direct_json(spec)
        else:
            raise ValueError(f"unknown run kind: {spec['kind']}")
        records.extend(run_records)
        components.extend(run_components)
    frame = pd.DataFrame(records)
    component_frame = pd.DataFrame(components)
    if not frame.empty:
        frame = frame.sort_values(["env", "method", "unit", "iteration"]).reset_index(drop=True)
    return frame, component_frame


def seed_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["env", "method", "run", "unit"], dropna=False)
    for (env, method, run, unit), group in grouped:
        group = group.sort_values("iteration")
        threshold = float(group["threshold"].iloc[0])
        best_idx = group["score"].idxmax()
        solved_group = group[group["score"] >= threshold]
        rows.append(
            {
                "env": env,
                "method": method,
                "run": run,
                "unit": unit,
                "initial_score": float(group.iloc[0]["score"]),
                "best_score": float(group.loc[best_idx, "score"]),
                "best_iteration": int(group.loc[best_idx, "iteration"]),
                "last_score": float(group.iloc[-1]["score"]),
                "iterations": int(group["iteration"].nunique()),
                "first_solved_iteration": int(solved_group.iloc[0]["iteration"]) if not solved_group.empty else np.nan,
                "solved": not solved_group.empty,
                "threshold": threshold,
                "normalized_initial": float(group.iloc[0]["score"]) / threshold,
                "normalized_best": float(group.loc[best_idx, "score"]) / threshold,
            }
        )
    return pd.DataFrame(rows)


def method_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (env, method), group in summary.groupby(["env", "method"]):
        rows.append(
            {
                "env": env,
                "method": method,
                "n": len(group),
                "solved": int(group["solved"].sum()),
                "solved_rate": float(group["solved"].mean()) if len(group) else float("nan"),
                "initial_mean": float(group["initial_score"].mean()),
                "initial_std": float(group["initial_score"].std(ddof=1)) if len(group) > 1 else float("nan"),
                "best_mean": float(group["best_score"].mean()),
                "best_std": float(group["best_score"].std(ddof=1)) if len(group) > 1 else float("nan"),
                "best_iteration_mean": float(group["best_iteration"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["env", "method"]).reset_index(drop=True)


def threshold_line(ax: plt.Axes, threshold: float, label: str | None = None) -> None:
    ax.axhline(threshold, color=COLORS["red"], lw=0.95, ls=(0, (4, 2)), alpha=0.9)
    if label:
        ax.text(0.98, threshold, label, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color=COLORS["red"])


def plot_env001_search(frame: pd.DataFrame, output_dir: Path) -> str | None:
    data = frame[(frame["env"] == "env_001") & (frame["method"] == "DERES")].copy()
    if data.empty:
        return None
    threshold = ENV_THRESHOLDS["env_001"]
    max_iter = int(data["iteration"].max())
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    cmap = plt.get_cmap("tab10")
    for idx, (unit, group) in enumerate(data.groupby("unit")):
        group = group.sort_values("iteration")
        color = cmap(idx % 10)
        axes[0].plot(group["iteration"], group["score"], marker="o", ms=3.1, lw=1.0, color=color, alpha=0.82, label=unit.replace("_", " "))
        axes[1].plot(group["iteration"], group["best_so_far"], marker="o", ms=3.0, lw=1.2, color=color, alpha=0.88)
    success_at = []
    for iteration in range(1, max_iter + 1):
        solved = 0
        for _, group in data.groupby("unit"):
            observed = group[group["iteration"] <= iteration]
            solved += int((observed["score"] >= threshold).any())
        success_at.append(solved)
    axes[2].step(range(1, max_iter + 1), success_at, where="post", color=COLORS["blue"], lw=1.8)
    axes[2].scatter(range(1, max_iter + 1), success_at, s=22, color=COLORS["blue"], edgecolor="white", linewidth=0.55)

    for ax, title in zip(axes[:2], ["Raw candidate scores", "Best-so-far scores"]):
        threshold_line(ax, threshold, "Solved threshold")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_xticks(range(1, max_iter + 1))
        clean_axis(ax)
    axes[0].set_ylabel("External evaluation return")
    axes[0].legend(frameon=False, ncol=2, loc="lower right")
    axes[2].set_title("Success@K")
    axes[2].set_xlabel("Iteration budget K")
    axes[2].set_ylabel("Solved seeds")
    axes[2].set_xticks(range(1, max_iter + 1))
    axes[2].set_ylim(-0.2, data["unit"].nunique() + 0.35)
    axes[2].set_yticks(range(0, data["unit"].nunique() + 1))
    clean_axis(axes[2])
    stem = "fig02_env001_search_trajectory"
    save_figure(fig, output_dir, stem)
    return stem


def plot_initial_to_best(summary: pd.DataFrame, output_dir: Path) -> str | None:
    data = summary[(summary["env"] == "env_001") & (summary["method"] == "DERES")].copy()
    if data.empty:
        return None
    threshold = ENV_THRESHOLDS["env_001"]
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for idx, row in data.sort_values("unit").iterrows():
        color = COLORS["green"] if row["best_score"] >= threshold else COLORS["gray_dark"]
        ax.plot([0, 1], [row["initial_score"], row["best_score"]], lw=1.2, color=color, alpha=0.75)
        ax.scatter([0, 1], [row["initial_score"], row["best_score"]], s=32, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(1.04, row["best_score"], row["unit"].replace("_", " "), va="center", fontsize=6.8, color=color)
    threshold_line(ax, threshold, "Solved threshold")
    ax.set_xlim(-0.15, 1.33)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Initial reward", "Best evolved reward"])
    ax.set_ylabel("External evaluation return")
    ax.set_title("Paired improvement on Env_001")
    clean_axis(ax)
    stem = "fig03_env001_initial_to_best"
    save_figure(fig, output_dir, stem)
    return stem


def plot_ablation_comparison(summary: pd.DataFrame, frame: pd.DataFrame, output_dir: Path) -> str | None:
    env_summary = summary[summary["env"] == "env_001"].copy()
    if env_summary.empty:
        return None
    rows: list[dict[str, Any]] = []
    deres = env_summary[env_summary["method"] == "DERES"]
    for _, row in deres.iterrows():
        rows.append({"method": "DERES best", "unit": row["unit"], "score": row["best_score"]})
        rows.append({"method": "LLM-once", "unit": row["unit"], "score": row["initial_score"]})
    without_hep = env_summary[env_summary["method"] == "w/o HEP"]
    for _, row in without_hep.iterrows():
        rows.append({"method": "w/o HEP", "unit": row["unit"], "score": row["best_score"]})
    independent = frame[(frame["env"] == "env_001") & (frame["method"] == "Budget-Matched Independent")]
    for _, row in independent.iterrows():
        rows.append({"method": "Independent-K", "unit": row["unit"], "score": row["score"]})
    data = pd.DataFrame(rows)
    if data.empty:
        return None
    order = ["LLM-once", "Independent-K", "w/o HEP", "DERES best"]
    order = [method for method in order if method in set(data["method"])]
    palette = {
        "LLM-once": COLORS["gray_dark"],
        "Independent-K": COLORS["amber"],
        "w/o HEP": COLORS["red"],
        "DERES best": COLORS["green"],
    }
    fig, ax = plt.subplots(figsize=(5.9, 3.05))
    rng = np.random.default_rng(12)
    for x, method in enumerate(order):
        values = data[data["method"] == method]["score"].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        ax.bar(x, float(np.mean(values)), width=0.58, color=palette[method], alpha=0.18, edgecolor=palette[method], linewidth=0.9)
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, s=36, color=palette[method], edgecolor="white", linewidth=0.65, zorder=3)
        if len(values) > 1:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            ax.errorbar(x, mean, yerr=std, fmt="none", color=palette[method], lw=1.0, capsize=3, zorder=2)
        ax.text(x, ax.get_ylim()[0] if False else min(values) - 18, f"n={len(values)}", ha="center", va="top", fontsize=7, color=COLORS["gray_dark"])
    threshold_line(ax, ENV_THRESHOLDS["env_001"], "Solved threshold")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=12, ha="right")
    ax.set_ylabel("External evaluation return")
    ax.set_title("Env_001 ablation comparison")
    clean_axis(ax)
    stem = "fig04_env001_ablation_comparison"
    save_figure(fig, output_dir, stem)
    return stem


def select_case_seed(frame: pd.DataFrame) -> str | None:
    data = frame[(frame["env"] == "env_001") & (frame["method"] == "DERES")]
    if data.empty:
        return None
    summary = seed_summary(data)
    late = summary.sort_values(["best_iteration", "best_score"], ascending=[False, False])
    return str(late.iloc[0]["unit"])


def plot_case_timeline(frame: pd.DataFrame, components: pd.DataFrame, output_dir: Path) -> str | None:
    unit = select_case_seed(frame)
    if not unit:
        return None
    data = frame[(frame["env"] == "env_001") & (frame["method"] == "DERES") & (frame["unit"] == unit)].sort_values("iteration")
    comp = components[(components["env"] == "env_001") & (components["method"] == "DERES")].copy()
    comp = comp[comp["seed"] == safe_float(unit.split("_")[-1])]
    if data.empty:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.15), gridspec_kw={"height_ratios": [1.0, 1.25]})
    axes[0].plot(data["iteration"], data["score"], marker="o", ms=4, lw=1.4, color=COLORS["blue"])
    axes[0].plot(data["iteration"], data["best_so_far"], marker="s", ms=3.2, lw=1.15, color=COLORS["green"], alpha=0.9, label="Best so far")
    threshold_line(axes[0], ENV_THRESHOLDS["env_001"], "Solved threshold")
    best = data.loc[data["score"].idxmax()]
    axes[0].scatter(best["iteration"], best["score"], s=58, color=COLORS["green"], edgecolor="white", linewidth=0.7, zorder=4)
    axes[0].annotate(f"best {best['score']:.1f}", (best["iteration"], best["score"]), xytext=(6, 6), textcoords="offset points", fontsize=7, color=COLORS["green"])
    axes[0].set_title(f"Reward evolution case: {unit.replace('_', ' ')}")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("External return")
    axes[0].legend(frameon=False, loc="lower right")
    clean_axis(axes[0])

    if comp.empty:
        axes[1].text(0.5, 0.5, "No component evidence available", ha="center", va="center", color=COLORS["gray_dark"])
        axes[1].axis("off")
    else:
        pivot = comp.pivot_table(index="component", columns="iteration", values="magnitude_share", aggfunc="first")
        pivot = pivot.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        top_components = pivot.max(axis=1).sort_values(ascending=False).head(8).index
        pivot = pivot.loc[top_components]
        image = axes[1].imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.01, float(pivot.values.max())))
        axes[1].set_yticks(range(len(pivot.index)))
        axes[1].set_yticklabels(pivot.index, fontsize=6.8)
        axes[1].set_xticks(range(len(pivot.columns)))
        axes[1].set_xticklabels([str(int(col)) for col in pivot.columns])
        axes[1].set_xlabel("Iteration")
        axes[1].set_title("Final-policy component magnitude share")
        axes[1].grid(False)
        cbar = fig.colorbar(image, ax=axes[1], fraction=0.025, pad=0.015)
        cbar.ax.tick_params(labelsize=6.7)
    stem = "fig05_env001_case_timeline"
    save_figure(fig, output_dir, stem)
    return stem


def plot_best_component_heatmap(summary: pd.DataFrame, components: pd.DataFrame, output_dir: Path) -> str | None:
    data = summary[(summary["env"] == "env_001") & (summary["method"] == "DERES")].copy()
    comp = components[(components["env"] == "env_001") & (components["method"] == "DERES")].copy()
    if data.empty or comp.empty:
        return None
    rows: list[dict[str, Any]] = []
    for _, seed_row in data.iterrows():
        seed = int(str(seed_row["unit"]).split("_")[-1])
        best_iter = int(seed_row["best_iteration"])
        sub = comp[(comp["seed"] == seed) & (comp["iteration"] == best_iter)]
        for _, row in sub.iterrows():
            rows.append(
                {
                    "case": f"seed {seed}\niter {best_iter}",
                    "component": row["component"],
                    "magnitude_share": safe_float(row["magnitude_share"], 0.0),
                }
            )
    matrix = pd.DataFrame(rows)
    if matrix.empty:
        return None
    pivot = matrix.pivot_table(index="case", columns="component", values="magnitude_share", aggfunc="first").fillna(0.0)
    top_cols = pivot.max(axis=0).sort_values(ascending=False).head(10).index
    pivot = pivot[top_cols]
    fig, ax = plt.subplots(figsize=(7.0, 2.95))
    image = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.01, float(pivot.values.max())))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_title("Component evidence at each seed's best reward")
    ax.grid(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    cbar.ax.tick_params(labelsize=6.8)
    stem = "fig06_env001_best_component_heatmap"
    save_figure(fig, output_dir, stem)
    return stem


def plot_cross_environment(summary: pd.DataFrame, output_dir: Path) -> str | None:
    data = summary[(summary["method"] == "DERES") & (summary["env"].isin(["env_001", "env_002", "env_005"]))].copy()
    data = data[(data["env"] != "env_005") | (data["unit"] == "seed_0")]
    if data.empty:
        return None
    order = ["env_001", "env_002", "env_005"]
    labels = {"env_001": "Env_001", "env_002": "Env_002", "env_005": "Env_005"}
    fig, ax = plt.subplots(figsize=(5.85, 3.0))
    rng = np.random.default_rng(3)
    for x, env in enumerate(order):
        env_data = data[data["env"] == env]
        if env_data.empty:
            continue
        for _, row in env_data.iterrows():
            jitter = rng.normal(0, 0.025)
            ax.plot([x - 0.16 + jitter, x + 0.16 + jitter], [row["normalized_initial"], row["normalized_best"]], color=COLORS["gray"], lw=0.8, alpha=0.55)
            ax.scatter(x - 0.16 + jitter, row["normalized_initial"], s=28, color=COLORS["gray_dark"], edgecolor="white", linewidth=0.55, zorder=3)
            ax.scatter(x + 0.16 + jitter, row["normalized_best"], s=34, color=COLORS["green"], edgecolor="white", linewidth=0.55, zorder=3)
        ax.text(x, max(env_data["normalized_best"].max(), env_data["normalized_initial"].max()) + 0.08, f"n={len(env_data)}", ha="center", fontsize=7, color=COLORS["gray_dark"])
    ax.axhline(1.0, color=COLORS["red"], lw=0.95, ls=(0, (4, 2)))
    ax.text(0.98, 1.0, "Solved threshold", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color=COLORS["red"])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels[env] for env in order])
    ax.set_ylabel("Score / solved threshold")
    ax.set_title("Cross-environment reward evolution")
    clean_axis(ax)
    ax.scatter([], [], s=30, color=COLORS["gray_dark"], edgecolor="white", linewidth=0.55, label="Initial")
    ax.scatter([], [], s=34, color=COLORS["green"], edgecolor="white", linewidth=0.55, label="Best")
    ax.legend(frameon=False, loc="upper left")
    stem = "fig07_cross_environment_generalization"
    save_figure(fig, output_dir, stem)
    return stem


def plot_env_understanding(frame: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> str | None:
    rows: list[dict[str, Any]] = []
    env2_summary = summary[summary["env"] == "env_002"]
    for method in ["Chat Direct", "Chat + Env Card", "Direct No Env Card", "No Expert Profile", "DeepSeek Direct", "DERES"]:
        method_summary_frame = env2_summary[env2_summary["method"] == method]
        if not method_summary_frame.empty:
            for _, row in method_summary_frame.iterrows():
                rows.append({"method": method, "score": row["best_score"], "unit": row["unit"]})
            continue
        direct = frame[(frame["env"] == "env_002") & (frame["method"] == method)]
        for _, row in direct.iterrows():
            rows.append({"method": method, "score": row["score"], "unit": row["unit"]})
    data = pd.DataFrame(rows)
    if data.empty:
        return None
    order = [method for method in ["Chat Direct", "Chat + Env Card", "Direct No Env Card", "No Expert Profile", "DeepSeek Direct", "DERES"] if method in set(data["method"])]
    colors = {
        "Chat Direct": COLORS["red"],
        "Chat + Env Card": COLORS["amber"],
        "Direct No Env Card": COLORS["gray_dark"],
        "No Expert Profile": COLORS["blue_dark"],
        "DeepSeek Direct": COLORS["green"],
        "DERES": COLORS["blue"],
    }
    fig, ax = plt.subplots(figsize=(7.05, 3.0))
    rng = np.random.default_rng(8)
    for x, method in enumerate(order):
        values = data[data["method"] == method]["score"].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        color = colors.get(method, COLORS["gray_dark"])
        ax.bar(x, float(np.mean(values)), width=0.56, color=color, alpha=0.18, edgecolor=color, linewidth=0.9)
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, s=34, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(x, min(values) - 20, f"n={len(values)}", ha="center", va="top", fontsize=7, color=COLORS["gray_dark"])
    threshold_line(ax, ENV_THRESHOLDS["env_002"], "Solved threshold")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=18, ha="right")
    ax.set_ylabel("External evaluation return")
    ax.set_title("Env_002 environment-understanding ablations")
    clean_axis(ax)
    stem = "fig08_env002_environment_understanding"
    save_figure(fig, output_dir, stem)
    return stem


def plot_method_success(frame: pd.DataFrame, output_dir: Path) -> str | None:
    env1 = frame[(frame["env"] == "env_001") & (frame["method"].isin(["DERES", "w/o HEP", "Budget-Matched Independent"]))].copy()
    if env1.empty:
        return None
    max_iter = int(env1["iteration"].max())
    fig, ax = plt.subplots(figsize=(4.6, 2.75))
    styles = {
        "DERES": (COLORS["green"], "-"),
        "w/o HEP": (COLORS["red"], "--"),
        "Budget-Matched Independent": (COLORS["amber"], ":"),
    }
    for method, (color, linestyle) in styles.items():
        method_data = env1[env1["method"] == method]
        if method_data.empty:
            continue
        units = sorted(method_data["unit"].unique())
        success = []
        for iteration in range(1, max_iter + 1):
            solved = 0
            for unit in units:
                observed = method_data[(method_data["unit"] == unit) & (method_data["iteration"] <= iteration)]
                solved += int((observed["score"] >= ENV_THRESHOLDS["env_001"]).any())
            success.append(solved / len(units) if units else np.nan)
        ax.plot(range(1, max_iter + 1), success, color=color, ls=linestyle, lw=1.7, marker="o", ms=3.3, label=f"{method} (n={len(units)})")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks(range(1, max_iter + 1))
    ax.set_xlabel("Candidate budget K")
    ax.set_ylabel("Success rate")
    ax.set_title("Budget-normalized search success on Env_001")
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)
    stem = "fig09_env001_success_by_budget"
    save_figure(fig, output_dir, stem)
    return stem


def write_tables(frame: pd.DataFrame, components: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary = seed_summary(frame)
    methods = method_summary(summary)
    frame.to_csv(table_dir / "all_iteration_results.csv", index=False)
    summary.to_csv(table_dir / "seed_summary.csv", index=False)
    methods.to_csv(table_dir / "method_summary.csv", index=False)
    if not components.empty:
        components.to_csv(table_dir / "component_evidence.csv", index=False)

    main_seed_rows = summary[
        ((summary["method"] == "DERES") & (summary["env"].isin(["env_001", "env_002"])))
        | ((summary["method"] == "DERES") & (summary["env"] == "env_005") & (summary["unit"] == "seed_0"))
    ]
    method_summary(main_seed_rows).to_csv(table_dir / "table_main_results.csv", index=False)
    ablation_rows = methods[
        ((methods["env"] == "env_001") & (methods["method"].isin(["DERES", "w/o HEP", "Budget-Matched Independent"])))
        | ((methods["env"] == "env_002") & (methods["method"].isin(["DERES", "Chat Direct", "Chat + Env Card", "Direct No Env Card", "No Expert Profile", "DeepSeek Direct"])))
    ]
    ablation_rows.to_csv(table_dir / "table_ablation_results.csv", index=False)
    return summary, methods


def copy_framework_reference(output_dir: Path) -> list[str]:
    copied: list[str] = []
    candidates = [
        ROOT / "figures/paper/framework/fig00_deres_agent_framework_generated.png",
        ROOT / "figures/paper/framework/fig02_deres_agent_framework_refined.png",
        ROOT / "figures/paper/framework/fig02_deres_agent_framework_refined.pdf",
        ROOT / "figures/paper/framework/fig02_deres_agent_framework_refined.svg",
    ]
    framework_dir = output_dir / "framework_reference"
    framework_dir.mkdir(parents=True, exist_ok=True)
    for path in candidates:
        if path.exists():
            destination = framework_dir / path.name
            shutil.copy2(path, destination)
            copied.append(str(destination.relative_to(ROOT)).replace("\\", "/"))
    return copied


def write_manifest(output_dir: Path, generated_stems: list[str], framework_refs: list[str], frame: pd.DataFrame, methods: pd.DataFrame) -> None:
    lines = [
        "# DERES Paper Figure Manifest",
        "",
        "Generated by `python -m analysis.paper.generate_deres_paper_figures`.",
        "",
        "## Figures",
        "",
    ]
    descriptions = {
        "fig01_deres_core_framework": "Main DERES framework diagram with the reward self-evolution agent highlighted as the core module.",
        "fig01_deres_framework_main": "Earlier DERES framework diagram kept only as a reference.",
        "fig01_deres_agentic_framework": "Alternative agentic DERES framework diagram.",
        "fig02_env001_search_trajectory": "Env_001 raw score, best-so-far score, and Success@K across five seeds.",
        "fig03_env001_initial_to_best": "Paired initial-to-best improvement on Env_001.",
        "fig04_env001_ablation_comparison": "Env_001 method comparison: LLM-once, budget-matched independent samples, w/o HEP, and DERES best.",
        "fig05_env001_case_timeline": "Representative Env_001 reward-evolution timeline with component evidence.",
        "fig06_env001_best_component_heatmap": "Component magnitude-share heatmap at each Env_001 seed's best reward.",
        "fig07_cross_environment_generalization": "Normalized initial-to-best improvement across Env_001, Env_002, and Env_005.",
        "fig08_env002_environment_understanding": "Env_002 ablations for environment understanding and model strength.",
        "fig09_env001_success_by_budget": "Budget-normalized Success@K comparison on Env_001.",
    }
    for stem in generated_stems:
        lines.append(f"- `{stem}.png` / `{stem}.pdf`: {descriptions.get(stem, '')}")
    lines.extend(["", "## Framework References", ""])
    if framework_refs:
        lines.extend(f"- `{path}`" for path in framework_refs)
    else:
        lines.append("- No framework reference figure found.")
    lines.extend(
        [
            "",
            "## Tables",
            "",
            "- `tables/all_iteration_results.csv`: all parsed reward-candidate evaluation records.",
            "- `tables/seed_summary.csv`: initial, best, solved status, and best iteration per seed/sample.",
            "- `tables/method_summary.csv`: mean/std and solved rate per method.",
            "- `tables/table_main_results.csv`: compact main-result table.",
            "- `tables/table_ablation_results.csv`: compact ablation table.",
            "- `tables/component_evidence.csv`: final-policy component evidence when available.",
            "",
            "## Data Notes",
            "",
            f"- Parsed evaluation records: {len(frame)}.",
            "- All main paper figures use external evaluation scores, not generated training rewards.",
            "- Best-so-far values are recomputed from observed evaluation scores.",
            "- Component evidence is diagnostic observed reward composition, not a causal contribution estimate.",
            "- Env_005 is treated as a seed_0 case study in main tables/figures; all parsed Env_005 records remain in `all_iteration_results.csv` and `method_summary.csv`.",
        ]
    )
    if not methods.empty:
        lines.extend(["", "## Method Summary Snapshot", ""])
        lines.extend(markdown_table(methods))
    (output_dir / "figure_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
        else:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[column]).replace("|", "/") for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DERES paper-ready figures from local experiment logs.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/paper/deres_main")
    args = parser.parse_args()

    apply_paper_style()
    frame, components = collect_all()
    if frame.empty:
        raise SystemExit("No experiment records found. Check RUN_SPECS paths.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, methods = write_tables(frame, components, args.output_dir)

    generated: list[str] = []
    if (args.output_dir / "fig01_deres_core_framework.png").exists():
        generated.append("fig01_deres_core_framework")
    for plotter in [
        lambda: plot_env001_search(frame, args.output_dir),
        lambda: plot_initial_to_best(summary, args.output_dir),
        lambda: plot_ablation_comparison(summary, frame, args.output_dir),
        lambda: plot_case_timeline(frame, components, args.output_dir),
        lambda: plot_best_component_heatmap(summary, components, args.output_dir),
        lambda: plot_cross_environment(summary, args.output_dir),
        lambda: plot_env_understanding(frame, summary, args.output_dir),
        lambda: plot_method_success(frame, args.output_dir),
    ]:
        stem = plotter()
        if stem:
            generated.append(stem)

    framework_refs = copy_framework_reference(args.output_dir)
    write_manifest(args.output_dir, generated, framework_refs, frame, methods)
    print(f"Generated {len(generated)} figure groups in {args.output_dir}")
    print(f"Wrote tables and manifest under {args.output_dir}")


if __name__ == "__main__":
    main()
