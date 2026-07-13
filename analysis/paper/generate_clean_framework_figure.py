from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent / "figures" / "framework"


def box(ax, x, y, w, h, title, subtitle="", face="#F7FAFC", edge="#2F3A45", lw=1.2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    title_fs = 9.4 if "\n" in title else 9.8
    subtitle_fs = 7.1 if subtitle.count("\n") >= 2 else 7.5
    ax.text(
        x + w / 2,
        y + h * 0.65,
        title,
        ha="center",
        va="center",
        fontsize=title_fs,
        fontweight="bold",
        color="#1D2733",
        linespacing=0.95,
        zorder=3,
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.34,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_fs,
            color="#4A5568",
            linespacing=1.18,
            zorder=3,
        )
    return patch


def arrow(ax, start, end, color="#334155", lw=1.7, rad=0.0, style="-|>", mutation_scale=13, dashed=False):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=(0, (4, 3)) if dashed else "solid",
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def generate():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
            "savefig.dpi": 420,
        }
    )

    fig, ax = plt.subplots(figsize=(13.6, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    colors = {
        "env": "#EEF4FA",
        "prior": "#F1F7EF",
        "gen": "#EBF3FF",
        "train": "#FFF4DE",
        "feedback": "#F4F6F8",
        "evolve": "#FFF0F0",
        "repair": "#F1ECFF",
        "memory": "#EAF7F7",
        "archive": "#F1F5F9",
        "accent": "#C2413D",
        "blue": "#2563A7",
        "green": "#2E7D4F",
        "amber": "#B7791F",
        "purple": "#6B4BB8",
        "gray": "#475569",
    }

    # Top input and generation path.
    box(ax, 0.045, 0.68, 0.145, 0.17, "Task\nAbstraction", "masked interface\n+ environment card", colors["env"], colors["gray"])
    box(ax, 0.235, 0.68, 0.15, 0.17, "Expert Design\nPriors", "task route, reward roles\noperator library", colors["prior"], colors["green"])
    box(ax, 0.43, 0.68, 0.15, 0.17, "Initial Reward\nSynthesis", "LLM writes executable\nreward code", colors["gen"], colors["blue"])

    # Training and evidence.
    box(ax, 0.635, 0.68, 0.15, 0.17, "Policy Training\n& Evaluation", "train on generated reward\nevaluate by external score", colors["train"], colors["amber"])
    box(ax, 0.635, 0.41, 0.15, 0.16, "Structured\nFeedback", "score, episode outcome\ncomponent statistics", colors["feedback"], colors["gray"])

    # Core evolution module.
    core = FancyBboxPatch(
        (0.205, 0.18),
        0.43,
        0.29,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.5,
        edgecolor=colors["accent"],
        facecolor="#FFF8F8",
        linestyle=(0, (5, 3)),
        zorder=0,
    )
    ax.add_patch(core)
    ax.text(
        0.42,
        0.445,
        "Structured Evolution Controller",
        ha="center",
        va="center",
        fontsize=10.3,
        fontweight="bold",
        color=colors["accent"],
    )
    box(ax, 0.235, 0.245, 0.155, 0.14, "Failure\nDiagnosis", "what failed?\nwhich signal?", "#FFFFFF", colors["accent"], lw=1.0)
    box(ax, 0.43, 0.235, 0.18, 0.16, "Attributable\nIntervention", "L1 scale repair\nL2 math transform\nL3 skeleton rebuild", "#FFFFFF", colors["accent"], lw=1.0)

    # Repair output.
    box(ax, 0.69, 0.245, 0.17, 0.14, "Local Reward\nRepair", "modify one target\ncomponent per round", colors["repair"], colors["purple"])

    # Memory/archive.
    box(ax, 0.08, 0.055, 0.18, 0.105, "Reward Memory", "history of designs, scores\nand intervention effects", colors["memory"], "#2F7F83", lw=1.0)
    box(ax, 0.70, 0.055, 0.18, 0.105, "Best Archive", "keeps best reward and policy\nfor rollback and reporting", colors["archive"], colors["gray"], lw=1.0)

    # Main flow.
    arrow(ax, (0.19, 0.765), (0.235, 0.765))
    arrow(ax, (0.385, 0.765), (0.43, 0.765))
    arrow(ax, (0.58, 0.765), (0.635, 0.765))
    arrow(ax, (0.71, 0.68), (0.71, 0.57), color=colors["amber"])
    arrow(ax, (0.635, 0.49), (0.60, 0.36), color=colors["accent"], rad=0.08)
    arrow(ax, (0.39, 0.315), (0.43, 0.315), color=colors["accent"])
    arrow(ax, (0.60, 0.315), (0.69, 0.315), color=colors["accent"])
    arrow(ax, (0.86, 0.315), (0.79, 0.68), color=colors["accent"], lw=2.4, rad=-0.42, mutation_scale=16)

    ax.text(
        0.90,
        0.53,
        "self-evolution loop",
        ha="center",
        va="center",
        fontsize=8.4,
        color=colors["accent"],
        rotation=72,
        fontweight="bold",
    )

    # Memory and archive links.
    arrow(ax, (0.17, 0.16), (0.31, 0.245), color="#2F7F83", dashed=True, lw=1.25, rad=-0.08, mutation_scale=10)
    arrow(ax, (0.50, 0.245), (0.22, 0.16), color="#2F7F83", dashed=True, lw=1.25, rad=-0.05, mutation_scale=10)
    arrow(ax, (0.71, 0.68), (0.79, 0.16), color=colors["gray"], dashed=True, lw=1.2, rad=0.05, mutation_scale=10)
    arrow(ax, (0.79, 0.16), (0.72, 0.245), color=colors["gray"], dashed=True, lw=1.2, rad=0.05, mutation_scale=10)

    # Ablation anchors.
    ablation_items = [
        ("w/o environment card", 0.117, 0.875),
        ("w/o expert priors", 0.31, 0.875),
        ("LLM-once", 0.505, 0.875),
        ("score-only feedback", 0.71, 0.615),
        ("w/o structured controller", 0.42, 0.125),
        ("w/o memory / archive", 0.50, 0.025),
    ]
    for label, x, y in ablation_items:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7.2,
            color="#64748B",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CBD5E1", linewidth=0.7),
        )

    ax.text(
        0.50,
        0.965,
        "DERES: Diagnosis-guided Expert Reward Evolution Search",
        ha="center",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color="#172033",
    )
    ax.text(
        0.50,
        0.925,
        "Failed rewards are reused as diagnostic evidence rather than discarded as unsuccessful samples.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#475569",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig01_deres_framework_clean.{suffix}", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    generate()
