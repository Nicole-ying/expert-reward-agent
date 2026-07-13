from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "paper" / "deres_main"

COLORS = {
    "ink": "#162033",
    "muted": "#526071",
    "line": "#CBD5E1",
    "panel": "#F8FAFC",
    "env": "#1F5F8B",
    "prior": "#2F7D55",
    "reward": "#6A55B5",
    "train": "#B7791F",
    "evidence": "#B54D44",
    "agent": "#1E7F83",
    "archive": "#475569",
    "soft_env": "#EAF3F8",
    "soft_prior": "#EDF7F1",
    "soft_reward": "#F0EEFB",
    "soft_train": "#FFF4DD",
    "soft_evidence": "#FFF0EE",
    "soft_agent": "#EAF7F7",
    "soft_archive": "#F1F5F9",
}


def rounded(ax, x, y, w, h, fc, ec, lw=1.35, r=0.022, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.010,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=None, lw=1.75, rad=0.0, dashed=False, ms=15, z=8):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color or COLORS["ink"],
            connectionstyle=f"arc3,rad={rad}",
            linestyle=(0, (4, 3)) if dashed else "solid",
            zorder=z,
        )
    )


def icon_doc(ax, cx, cy, s, color):
    ax.add_patch(Rectangle((cx - s * 0.28, cy - s * 0.34), s * 0.48, s * 0.68, fill=False, ec=color, lw=1.6, zorder=6))
    ax.add_patch(Polygon([[cx + s * 0.20, cy + s * 0.34], [cx + s * 0.34, cy + s * 0.20], [cx + s * 0.20, cy + s * 0.20]], fill=False, ec=color, lw=1.5, zorder=6))
    for i in range(3):
        ax.plot([cx - s * 0.18, cx + s * 0.12], [cy + s * (0.08 - i * 0.14), cy + s * (0.08 - i * 0.14)], color=color, lw=1.15, zorder=6)


def icon_brain(ax, cx, cy, s, color):
    for dx, dy, rr in [(-0.15, 0.10, 0.15), (0.06, 0.14, 0.18), (-0.02, -0.08, 0.20), (0.19, -0.06, 0.14)]:
        ax.add_patch(Circle((cx + s * dx, cy + s * dy), s * rr, fill=False, ec=color, lw=1.55, zorder=6))
    ax.plot([cx - s * 0.10, cx + s * 0.24], [cy - s * 0.25, cy - s * 0.31], color=color, lw=1.4, zorder=6)


def icon_card(ax, cx, cy, s, color):
    rounded(ax, cx - s * 0.36, cy - s * 0.27, s * 0.72, s * 0.54, fc="white", ec=color, lw=1.5, r=0.005, z=6)
    ax.plot([cx - s * 0.25, cx + s * 0.25], [cy + s * 0.10, cy + s * 0.10], color=color, lw=1.15, zorder=6)
    ax.plot([cx - s * 0.25, cx + s * 0.15], [cy - s * 0.05, cy - s * 0.05], color=color, lw=1.15, zorder=6)
    ax.plot([cx - s * 0.25, cx + s * 0.22], [cy - s * 0.20, cy - s * 0.20], color=color, lw=1.15, zorder=6)


def icon_priors(ax, cx, cy, s, color):
    pts = [(cx, cy + s * 0.30), (cx - s * 0.28, cy - s * 0.16), (cx + s * 0.28, cy - s * 0.16), (cx, cy - s * 0.02)]
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)]:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=color, lw=1.25, zorder=6)
    for px, py in pts:
        ax.add_patch(Circle((px, py), s * 0.055, fc="white", ec=color, lw=1.45, zorder=7))


def icon_code(ax, cx, cy, s, color):
    rounded(ax, cx - s * 0.38, cy - s * 0.28, s * 0.76, s * 0.56, fc="white", ec=color, lw=1.5, r=0.006, z=6)
    ax.text(cx, cy, "</>", ha="center", va="center", fontsize=10.5, color=color, fontweight="bold", zorder=7)
    for i in range(3):
        ax.add_patch(Circle((cx - s * 0.27 + i * s * 0.09, cy + s * 0.18), s * 0.018, color=color, zorder=7))


def icon_train(ax, cx, cy, s, color):
    for i, height in enumerate([0.28, 0.45, 0.64]):
        ax.add_patch(Rectangle((cx - s * 0.29 + i * s * 0.21, cy - s * 0.30), s * 0.10, s * height, fc=color, ec=color, zorder=6))
    ax.plot([cx - s * 0.35, cx + s * 0.35], [cy - s * 0.30, cy - s * 0.30], color=color, lw=1.35, zorder=6)


def icon_evidence(ax, cx, cy, s, color):
    rounded(ax, cx - s * 0.34, cy - s * 0.30, s * 0.68, s * 0.60, fc="white", ec=color, lw=1.5, r=0.004, z=6)
    for t in [-0.10, 0.10]:
        ax.plot([cx - s * 0.34, cx + s * 0.34], [cy + s * t, cy + s * t], color=color, lw=1.0, zorder=6)
    for t in [-0.11, 0.11]:
        ax.plot([cx + s * t, cx + s * t], [cy - s * 0.30, cy + s * 0.30], color=color, lw=1.0, zorder=6)


def icon_agent(ax, cx, cy, s, color):
    ax.plot([cx, cx], [cy + s * 0.30, cy - s * 0.04], color=color, lw=1.55, zorder=6)
    ax.plot([cx, cx - s * 0.27], [cy - s * 0.04, cy - s * 0.28], color=color, lw=1.55, zorder=6)
    ax.plot([cx, cx + s * 0.27], [cy - s * 0.04, cy - s * 0.28], color=color, lw=1.55, zorder=6)
    for px, py in [(cx, cy + s * 0.31), (cx, cy - s * 0.04), (cx - s * 0.29, cy - s * 0.30), (cx + s * 0.29, cy - s * 0.30)]:
        ax.add_patch(Circle((px, py), s * 0.052, fc="white", ec=color, lw=1.45, zorder=7))


def icon_archive(ax, cx, cy, s, color):
    ax.add_patch(Rectangle((cx - s * 0.31, cy - s * 0.22), s * 0.62, s * 0.44, fill=False, ec=color, lw=1.55, zorder=6))
    ax.plot([cx - s * 0.22, cx + s * 0.22], [cy + s * 0.12, cy + s * 0.12], color=color, lw=1.0, zorder=7)
    ax.plot([cx - s * 0.14, cx + s * 0.14], [cy - s * 0.03, cy - s * 0.03], color=color, lw=1.0, zorder=7)


def node(ax, x, y, w, h, title, subtitle, color, face, icon_fn):
    rounded(ax, x, y, w, h, fc=face, ec=color, lw=1.25, r=0.018)
    icon_fn(ax, x + w * 0.13, y + h * 0.52, min(w, h) * 0.40, color)
    title_fs = 7.7 if "\n" in title or len(title) > 18 else 8.2
    subtitle_fs = 5.85 if subtitle.count("\n") >= 1 else 6.1
    ax.text(
        x + w * 0.34,
        y + h * 0.66,
        title,
        ha="left",
        va="center",
        fontsize=title_fs,
        fontweight="bold",
        color=COLORS["ink"],
        linespacing=0.96,
        zorder=7,
    )
    ax.text(
        x + w * 0.34,
        y + h * 0.36,
        subtitle,
        ha="left",
        va="center",
        fontsize=subtitle_fs,
        color=COLORS["muted"],
        linespacing=1.08,
        zorder=7,
    )


def small_chip(ax, x, y, text, color):
    rounded(ax, x, y, 0.124, 0.036, fc="white", ec=color, lw=0.85, r=0.008, z=8)
    ax.text(x + 0.062, y + 0.018, text, ha="center", va="center", fontsize=7.2, color=color, zorder=9)


def generate() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 160,
            "savefig.dpi": 430,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(14.2, 7.0))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Light lanes.
    rounded(ax, 0.035, 0.675, 0.93, 0.205, fc="#FBFCFE", ec="#E2E8F0", lw=0.7, r=0.018, z=0)
    rounded(ax, 0.035, 0.385, 0.93, 0.205, fc="#FBFCFE", ec="#E2E8F0", lw=0.7, r=0.018, z=0)
    rounded(ax, 0.035, 0.095, 0.93, 0.205, fc="#FBFCFE", ec="#E2E8F0", lw=0.7, r=0.018, z=0)
    ax.text(0.055, 0.855, "Task perception", fontsize=8.1, color=COLORS["muted"], fontweight="bold")
    ax.text(0.055, 0.565, "Reward proposal and policy feedback", fontsize=8.1, color=COLORS["muted"], fontweight="bold")
    ax.text(0.055, 0.275, "Diagnostic self-evolution", fontsize=8.1, color=COLORS["muted"], fontweight="bold")

    # Top lane: environment understanding.
    node(ax, 0.070, 0.705, 0.215, 0.125, "Masked\nenvironment", "task text, obs/action space\nmasked step interface", COLORS["env"], COLORS["soft_env"], icon_doc)
    node(ax, 0.382, 0.705, 0.235, 0.125, "Environment\nUnderstanding LLM", "task goal, variable semantics\nusable signals, failure modes", COLORS["env"], COLORS["soft_env"], icon_brain)
    node(ax, 0.715, 0.705, 0.215, 0.125, "Environment\ncard", "compact task model for\nreward design", COLORS["env"], COLORS["soft_env"], icon_card)
    arrow(ax, (0.285, 0.767), (0.382, 0.767), color=COLORS["env"])
    arrow(ax, (0.617, 0.767), (0.715, 0.767), color=COLORS["env"])
    small_chip(ax, 0.118, 0.662, "w/o Env LLM", COLORS["env"])

    # Middle lane.
    node(ax, 0.070, 0.415, 0.215, 0.125, "Expert design\npriors", "reward roles, operators\nrisk checks, task families", COLORS["prior"], COLORS["soft_prior"], icon_priors)
    node(ax, 0.335, 0.415, 0.210, 0.125, "Initial Reward\nSynthesis", "compose signals into\nexecutable reward R_t", COLORS["reward"], COLORS["soft_reward"], icon_code)
    node(ax, 0.600, 0.415, 0.185, 0.125, "Policy\nTraining", "train policy on R_t\nwith PPO", COLORS["train"], COLORS["soft_train"], icon_train)
    node(ax, 0.825, 0.415, 0.130, 0.125, "External\nscore", "task return\nJ_env", COLORS["train"], COLORS["soft_train"], icon_train)
    arrow(ax, (0.285, 0.477), (0.335, 0.477), color=COLORS["prior"])
    arrow(ax, (0.545, 0.477), (0.600, 0.477), color=COLORS["reward"])
    arrow(ax, (0.785, 0.477), (0.825, 0.477), color=COLORS["train"])
    arrow(ax, (0.825, 0.705), (0.447, 0.540), color=COLORS["env"], dashed=True, rad=0.03, ms=12, lw=1.15, z=3)
    small_chip(ax, 0.378, 0.372, "LLM-once", COLORS["reward"])

    # Bottom lane.
    node(ax, 0.070, 0.125, 0.230, 0.125, "Reward Evidence\nSystem", "score, episode outcome\ncomponent stats, memory", COLORS["evidence"], COLORS["soft_evidence"], icon_evidence)
    node(ax, 0.385, 0.125, 0.245, 0.125, "Hierarchical\nEvolution Agent", "diagnose failure, choose\nL1 scale / L2 form / L3 rebuild", COLORS["agent"], COLORS["soft_agent"], icon_agent)
    node(ax, 0.705, 0.125, 0.195, 0.125, "Local Reward\nRepair", "modify one target component\nor rebuild skeleton", COLORS["agent"], COLORS["soft_agent"], icon_code)
    node(ax, 0.795, 0.260, 0.170, 0.095, "Best Archive", "best reward and policy", COLORS["archive"], COLORS["soft_archive"], icon_archive)

    arrow(ax, (0.890, 0.415), (0.230, 0.250), color=COLORS["evidence"], rad=-0.10, ms=16, lw=1.75, z=3)
    arrow(ax, (0.300, 0.187), (0.385, 0.187), color=COLORS["evidence"])
    arrow(ax, (0.630, 0.187), (0.705, 0.187), color=COLORS["agent"])
    arrow(ax, (0.810, 0.250), (0.850, 0.260), color=COLORS["archive"], dashed=True, ms=12, lw=1.10, z=4)

    # Main reward loop from repair back to reward synthesis, routed through empty space.
    ax.plot([0.802, 0.802, 0.445, 0.445], [0.125, 0.073, 0.073, 0.415], color=COLORS["agent"], lw=1.8, zorder=1)
    arrow(ax, (0.445, 0.398), (0.445, 0.415), color=COLORS["agent"], ms=15, lw=1.8, z=3)
    ax.text(0.615, 0.057, "next reward candidate", ha="center", va="center", fontsize=7.8, color=COLORS["agent"], fontweight="bold")

    # Best archive and memory connections.
    arrow(ax, (0.530, 0.250), (0.835, 0.300), color=COLORS["archive"], dashed=True, rad=0.08, ms=12, lw=1.10, z=3)
    arrow(ax, (0.190, 0.125), (0.190, 0.250), color=COLORS["evidence"], dashed=True, rad=0.0, ms=12, lw=1.15)
    small_chip(ax, 0.132, 0.082, "score-only feedback", COLORS["evidence"])
    small_chip(ax, 0.442, 0.082, "w/o L1/L2/L3", COLORS["agent"])

    OUT.mkdir(parents=True, exist_ok=True)
    stem = "fig01_deres_agentic_framework"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


if __name__ == "__main__":
    generate()
