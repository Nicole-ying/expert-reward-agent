"""Generate redesigned candidate figures for the ISCSIC 2026 CREATE paper.

This script does not overwrite the submitted figures by default. It creates two
candidate replacements in ../figures/:

  - fig3_mechanism_ablation_redesign.pdf/png
  - fig4_repair_mechanism_schematic.pdf/png

Run from this directory or from the script directory:

    python scripts/redraw_paper_figures.py

The Fig. 3 data are taken directly from Table 1 of the draft. Fig. 4 is a
schematic event-flow summary of the two representative repair cases described
in Section "How Evidence Becomes an Edit"; it is intentionally not a fabricated
per-round time series.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_all(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)


def make_fig3_ablation() -> None:
    conditions = [
        "Independent-10",
        "Stateless\nrevision",
        "CREATE w/o\nevidence",
        "CREATE w/o\nbounded edit",
        "CREATE",
    ]
    means = np.array([-0.74, 44.00, 134.97, 114.21, 228.98])
    stds = np.array([114.40, 65.90, 148.43, 46.31, 16.54])
    solved = ["0/5", "0/5", "2/5", "0/5", "5/5"]

    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    bars = ax.bar(x, means, yerr=stds, capsize=3, linewidth=0.8, edgecolor="black")
    bars[-1].set_linewidth(1.6)

    ax.axhline(200, linestyle="--", linewidth=1.0, color="black")
    ax.text(len(conditions) - 0.25, 205, "solved threshold", ha="right", va="bottom", fontsize=8)

    for i, (m, s) in enumerate(zip(means, solved)):
        y = m + stds[i] + 12 if m >= 0 else 12
        ax.text(i, y, s, ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Best search fitness")
    ax.set_xlabel("Reward-search condition")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_title("Mechanism ablation under matched reward-evaluation budget", fontsize=10)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-180, 285)
    fig.tight_layout()
    save_all(fig, "fig3_mechanism_ablation_redesign")
    plt.close(fig)


def make_fig4_schematic() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.axis("off")

    rows = [0.72, 0.30]
    labels = ["Seed 0: collapse and recovery", "Seed 3: progressive convergence"]
    stages = [
        [
            ("Exploratory edits", "scale / sparse-dense / form changes"),
            ("Collapse", "progress signal penalizes stability"),
            ("Diagnosis", "unstable reference frame"),
            ("L2 repair", "restore state-based signal + altitude factor"),
            ("Solved", "fitness rises above threshold"),
        ],
        [
            ("Initial failure", "absolute-distance term weakly directional"),
            ("L2 refactor", "potential-difference approach shaping"),
            ("Regression", "over-restrictive contact adjustment"),
            ("Memory comparison", "localize most recent semantic change"),
            ("Solved", "reachable single-to-double contact gradient"),
        ],
    ]

    box_w = 0.17
    box_h = 0.16
    x0 = 0.08
    gap = 0.025

    for r, y in enumerate(rows):
        ax.text(0.01, y + 0.03, labels[r], ha="left", va="center", fontsize=9, fontweight="bold")
        for i, (title, desc) in enumerate(stages[r]):
            x = x0 + i * (box_w + gap)
            rect = plt.Rectangle((x, y - box_h / 2), box_w, box_h, fill=False, linewidth=1.0)
            ax.add_patch(rect)
            ax.text(x + box_w / 2, y + 0.025, title, ha="center", va="center", fontsize=8, fontweight="bold")
            ax.text(x + box_w / 2, y - 0.035, desc, ha="center", va="center", fontsize=6.8, wrap=True)
            if i < len(stages[r]) - 1:
                ax.annotate(
                    "",
                    xy=(x + box_w + gap * 0.75, y),
                    xytext=(x + box_w, y),
                    arrowprops=dict(arrowstyle="->", linewidth=0.9),
                )

    ax.text(
        0.5,
        0.05,
        "Evidence organizer identifies the principal semantic failure; bounded editing tests one repair hypothesis; memory prevents regression.",
        ha="center",
        va="center",
        fontsize=8,
    )
    fig.tight_layout()
    save_all(fig, "fig4_repair_mechanism_schematic")
    plt.close(fig)


def main() -> None:
    make_fig3_ablation()
    make_fig4_schematic()
    print(f"Saved redesigned figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
