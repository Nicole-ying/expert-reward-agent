from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.paper.paper_style import COLORS, apply_paper_style, clean_axis, save_figure


ROOT = Path(__file__).resolve().parents[2]
THRESHOLD = 200.0
SEED_COLORS = plt.get_cmap("tab10")


def threshold_line(ax) -> None:
    ax.axhline(THRESHOLD, color=COLORS["red"], lw=0.9, ls=(0, (4, 2)))
    ax.text(0.99, THRESHOLD + 5, "Solved threshold (200)", transform=ax.get_yaxis_transform(), ha="right", color=COLORS["red"], fontsize=7)


def plot_search_dynamics(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), sharex=True, sharey=True)
    for seed, group in frame.groupby("seed"):
        color = SEED_COLORS(int(seed) % 10)
        group = group.sort_values("iteration")
        grid = group.set_index("iteration").reindex(range(1, 11))
        axes[0].plot(grid.index, grid["score"], marker="o", ms=3.2, lw=1.1, color=color, alpha=0.85, label=f"Seed {seed}")
        axes[1].plot(grid.index, grid["best_so_far"], marker="o", ms=3.0, lw=1.25, color=color, alpha=0.9)
        for _, row in group[group["restart_scheduled"]].iterrows():
            axes[0].scatter(row["iteration"], row["score"], marker="D", s=30, facecolors="none", edgecolors=color, linewidths=1.0, zorder=4)
    for ax, title in zip(axes, ["Observed reward candidates", "Best reward found so far"]):
        threshold_line(ax)
        ax.set_title(title)
        ax.set_xlabel("Reward-search iteration")
        ax.set_xticks(range(1, 11))
        clean_axis(ax)
    axes[0].set_ylabel("External evaluation return")
    axes[0].legend(frameon=False, ncol=2, loc="lower right")
    axes[0].text(0.02, 0.03, "Open diamond: restart scheduled", transform=axes[0].transAxes, fontsize=6.7, color=COLORS["gray_dark"])
    save_figure(fig, output_dir, "fig01_reward_search_dynamics")


def plot_initial_to_best(frame: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for seed, group in frame.groupby("seed"):
        group = group.sort_values("iteration")
        rows.append((int(seed), float(group.iloc[0]["score"]), float(group["score"].max())))
    fig, ax = plt.subplots(figsize=(3.45, 2.85))
    for seed, initial, best in rows:
        color = COLORS["green"] if best >= THRESHOLD else COLORS["gray_dark"]
        ax.plot([0, 1], [initial, best], color=color, lw=1.15, alpha=0.72)
        ax.scatter([0, 1], [initial, best], color=color, s=28, edgecolor="white", linewidth=0.55, zorder=3)
    label_rows = sorted([(seed, best) for seed, _, best in rows], key=lambda item: item[1])
    label_positions = []
    for _, best in label_rows:
        label_positions.append(max(best, label_positions[-1] + 12 if label_positions else best))
    for (seed, best), label_y in zip(label_rows, label_positions):
        color = COLORS["green"] if best >= THRESHOLD else COLORS["gray_dark"]
        ax.plot([1.01, 1.055], [best, label_y], color=color, lw=0.65, alpha=0.65)
        ax.text(1.065, label_y, f"Seed {seed}", va="center", fontsize=6.7, color=color)
    ax.axhline(THRESHOLD, color=COLORS["red"], lw=0.9, ls=(0, (4, 2)))
    ax.text(0.03, THRESHOLD + 5, "Solved threshold (200)", color=COLORS["red"], fontsize=7)
    ax.set_xlim(-0.18, 1.34)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Initial reward", "Best evolved reward"])
    ax.set_ylabel("External evaluation return")
    ax.set_title("Paired improvement within each search seed")
    clean_axis(ax)
    save_figure(fig, output_dir, "fig02_initial_to_best_improvement")


def plot_search_efficiency(frame: pd.DataFrame, output_dir: Path) -> None:
    seeds = sorted(frame["seed"].unique())
    success_at = []
    for iteration in range(1, 11):
        solved = 0
        for seed in seeds:
            observed = frame[(frame["seed"] == seed) & (frame["iteration"] <= iteration)]
            solved += int((observed["score"] >= THRESHOLD).any())
        success_at.append(solved / len(seeds))
    first = []
    unsolved = []
    for seed, group in frame.groupby("seed"):
        solved = group[group["score"] >= THRESHOLD]
        if solved.empty:
            unsolved.append(int(seed))
        else:
            first.append((int(seed), int(solved["iteration"].min())))

    fig, axes = plt.subplots(1, 2, figsize=(6.55, 2.65))
    axes[0].step(range(1, 11), success_at, where="post", color=COLORS["blue"], lw=1.8)
    axes[0].scatter(range(1, 11), success_at, s=22, color=COLORS["blue"], edgecolor="white", linewidth=0.5)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xticks(range(1, 11))
    axes[0].set_xlabel("Iteration budget K")
    axes[0].set_ylabel("Success@K across seeds")
    axes[0].set_title("Cumulative search success")

    for index, (seed, iteration) in enumerate(first):
        axes[1].scatter(iteration, index, s=38, color=COLORS["green"], edgecolor="white", linewidth=0.6)
        axes[1].text(iteration + 0.15, index, f"Seed {seed}", va="center", fontsize=7)
    for offset, seed in enumerate(unsolved, start=len(first)):
        axes[1].scatter(10, offset, marker="x", s=38, color=COLORS["red"], linewidth=1.2)
        axes[1].text(9.8, offset, f"Seed {seed} unsolved", ha="right", va="center", fontsize=7, color=COLORS["red"])
    axes[1].set_xlim(0.5, 10.7)
    axes[1].set_xticks(range(1, 11))
    axes[1].set_yticks([])
    axes[1].set_xlabel("First solved iteration")
    axes[1].set_title("Time to first solved reward")
    for ax in axes:
        clean_axis(ax)
    save_figure(fig, output_dir, "fig03_search_efficiency")


def plot_case_studies(frame: pd.DataFrame, components: pd.DataFrame, output_dir: Path) -> None:
    case_seeds = [4, 5]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.35), gridspec_kw={"height_ratios": [1.0, 1.15]})
    for column, seed in enumerate(case_seeds):
        group = frame[frame["seed"] == seed].sort_values("iteration")
        ax = axes[0, column]
        ax.plot(group["iteration"], group["score"], marker="o", ms=4, lw=1.5, color=COLORS["blue"])
        ax.axhline(THRESHOLD, color=COLORS["red"], lw=0.9, ls=(0, (4, 2)))
        best = group.loc[group["score"].idxmax()]
        ax.scatter(best["iteration"], best["score"], s=48, color=COLORS["green"], edgecolor="white", linewidth=0.7, zorder=4)
        ax.annotate(f"best {best['score']:.1f}", (best["iteration"], best["score"]), xytext=(5, 6), textcoords="offset points", fontsize=7, color=COLORS["green"])
        ax.margins(y=0.12)
        ax.set_title(f"Seed {seed}: external return")
        ax.set_xlabel("Iteration")
        if column == 0:
            ax.set_ylabel("Return")
        clean_axis(ax)

        comp = components[components["seed"] == seed].copy()
        comp = comp[comp["iteration"].isin(group["iteration"])]
        pivot = comp.pivot_table(index="component", columns="iteration", values="mean", aggfunc="first")
        if not pivot.empty:
            scale = pivot.abs().max(axis=0).replace(0, 1.0)
            normalized = pivot.divide(scale, axis=1).fillna(0.0)
            axes[1, column].imshow(normalized.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
            axes[1, column].set_yticks(range(len(normalized.index)))
            axes[1, column].set_yticklabels(normalized.index, fontsize=6.7)
            axes[1, column].set_xticks(range(len(normalized.columns)))
            axes[1, column].set_xticklabels(normalized.columns)
            axes[1, column].set_xlabel("Iteration")
            axes[1, column].set_title("Signed component mean (column-normalized)")
            axes[1, column].grid(False)
    save_figure(fig, output_dir, "fig04_diagnostic_case_studies")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/paper/v6_exploratory")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/paper/v6_exploratory")
    args = parser.parse_args()
    apply_paper_style()
    frame = pd.read_csv(args.data_dir / "iteration_results.csv")
    components = pd.read_csv(args.data_dir / "component_means.csv")
    plot_search_dynamics(frame, args.output_dir)
    plot_initial_to_best(frame, args.output_dir)
    plot_search_efficiency(frame, args.output_dir)
    plot_case_studies(frame, components, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
