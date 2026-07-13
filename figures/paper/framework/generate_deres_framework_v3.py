from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT = Path(__file__).resolve().parent

COLORS = {
    "navy": "#0B345D",
    "blue": "#135EAA",
    "teal": "#087B7A",
    "amber": "#B87405",
    "green": "#237A1E",
    "ink": "#111827",
    "muted": "#4B5563",
    "line": "#CAD2DB",
    "panel": "#FCFDFE",
    "red": "#D61F2C",
    "black": "#1F2937",
}


def rounded(ax, x, y, w, h, fc="white", ec="#222", lw=1.5, r=0.016, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color="#222", lw=2.5, ms=22, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=10,
        )
    )


def panel(ax, x, y, w, h, color, title):
    rounded(ax, x, y, w, h, fc=COLORS["panel"], ec=color, lw=1.6, r=0.018, z=1)
    hh = 0.095 * h
    ax.add_patch(Rectangle((x, y + h - hh), w, hh, facecolor=color, edgecolor=color, zorder=2))
    ax.text(
        x + 0.018 * w,
        y + h - hh / 2,
        title,
        ha="left",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color="white",
        zorder=3,
    )
    return hh


def icon_env(ax, cx, cy, s, c):
    ax.add_patch(Circle((cx, cy), s * 0.42, fill=False, ec=c, lw=1.7, zorder=5))
    ax.plot([cx - s * 0.38, cx + s * 0.38], [cy, cy], color=c, lw=1.3, zorder=5)
    ax.plot([cx, cx], [cy - s * 0.38, cy + s * 0.38], color=c, lw=1.3, zorder=5)
    ax.add_patch(FancyBboxPatch((cx - s * 0.22, cy - s * 0.39), s * 0.44, s * 0.78, boxstyle="round,pad=0,rounding_size=0.05", fill=False, ec=c, lw=1.1, zorder=5))


def icon_brain(ax, cx, cy, s, c):
    for dx, dy, r in [(-0.14, 0.10, 0.17), (0.08, 0.15, 0.20), (0.00, -0.09, 0.20)]:
        ax.add_patch(Circle((cx + s * dx, cy + s * dy), s * r, fill=False, ec=c, lw=1.7, zorder=5))
    ax.plot([cx - s * 0.04, cx + s * 0.25], [cy - s * 0.23, cy - s * 0.34], color=c, lw=1.5, zorder=5)


def icon_doc(ax, cx, cy, s, c):
    ax.add_patch(Rectangle((cx - s * 0.28, cy - s * 0.34), s * 0.48, s * 0.68, fill=False, ec=c, lw=1.8, zorder=5))
    ax.add_patch(Polygon([[cx + s * 0.20, cy + s * 0.34], [cx + s * 0.34, cy + s * 0.20], [cx + s * 0.20, cy + s * 0.20]], fill=False, ec=c, lw=1.8, zorder=5))
    for i in range(3):
        ax.plot([cx - s * 0.18, cx + s * 0.12], [cy + s * (0.08 - 0.14 * i), cy + s * (0.08 - 0.14 * i)], color=c, lw=1.3, zorder=5)


def icon_plan(ax, cx, cy, s, c):
    ax.add_patch(Rectangle((cx - s * 0.36, cy - s * 0.28), s * 0.72, s * 0.56, fill=False, ec=c, lw=1.8, zorder=5))
    ax.plot([cx - s * 0.24, cx + s * 0.24], [cy + s * 0.10, cy + s * 0.10], color=c, lw=1.4, zorder=5)
    ax.plot([cx - s * 0.24, cx + s * 0.08], [cy - s * 0.05, cy - s * 0.05], color=c, lw=1.4, zorder=5)
    ax.plot([cx - s * 0.24, cx + s * 0.18], [cy - s * 0.20, cy - s * 0.20], color=c, lw=1.4, zorder=5)


def icon_code(ax, cx, cy, s, c):
    rounded(ax, cx - s * 0.40, cy - s * 0.30, s * 0.80, s * 0.60, fc="none", ec=c, lw=1.8, r=0.004, z=5)
    ax.text(cx, cy, "</>", ha="center", va="center", fontsize=12, fontweight="bold", color=c, zorder=6)
    for i in range(3):
        ax.add_patch(Circle((cx - s * 0.28 + i * s * 0.09, cy + s * 0.20), s * 0.018, color=c, zorder=6))


def icon_shield(ax, cx, cy, s, c):
    pts = [(cx, cy + s * 0.38), (cx + s * 0.34, cy + s * 0.24), (cx + s * 0.26, cy - s * 0.20), (cx, cy - s * 0.38), (cx - s * 0.26, cy - s * 0.20), (cx - s * 0.34, cy + s * 0.24)]
    ax.add_patch(Polygon(pts, fill=False, ec=c, lw=1.8, zorder=5))
    ax.plot([cx - s * 0.15, cx - s * 0.02, cx + s * 0.22], [cy, cy - s * 0.13, cy + s * 0.12], color=c, lw=1.8, zorder=5)


def icon_network(ax, cx, cy, s, c):
    pts = [(cx - s * 0.28, cy - s * 0.18), (cx, cy + s * 0.32), (cx + s * 0.30, cy - s * 0.14), (cx, cy - s * 0.02)]
    for a, b in [(0, 1), (1, 2), (0, 3), (3, 2), (1, 3)]:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=c, lw=1.4, zorder=5)
    for px, py in pts:
        ax.add_patch(Circle((px, py), s * 0.065, fc="white", ec=c, lw=1.7, zorder=6))


def icon_bars(ax, cx, cy, s, c):
    for i, ht in enumerate([0.25, 0.42, 0.62]):
        ax.add_patch(Rectangle((cx - s * 0.28 + i * s * 0.20, cy - s * 0.30), s * 0.10, s * ht, facecolor=c, edgecolor=c, zorder=5))
    ax.plot([cx - s * 0.34, cx + s * 0.34], [cy - s * 0.30, cy - s * 0.30], color=c, lw=1.5, zorder=5)


def icon_search(ax, cx, cy, s, c):
    ax.add_patch(Circle((cx - s * 0.08, cy + s * 0.08), s * 0.22, fill=False, ec=c, lw=1.8, zorder=5))
    ax.plot([cx + s * 0.08, cx + s * 0.32], [cy - s * 0.10, cy - s * 0.32], color=c, lw=2.0, zorder=5)


def icon_db(ax, cx, cy, s, c):
    ax.add_patch(FancyBboxPatch((cx - s * 0.32, cy + s * 0.12), s * 0.64, s * 0.18, boxstyle="round,pad=0,rounding_size=0.045", fill=False, ec=c, lw=1.7, zorder=5))
    ax.add_patch(Rectangle((cx - s * 0.32, cy - s * 0.20), s * 0.64, s * 0.40, fill=False, ec=c, lw=1.7, zorder=5))
    ax.add_patch(FancyBboxPatch((cx - s * 0.32, cy - s * 0.30), s * 0.64, s * 0.18, boxstyle="round,pad=0,rounding_size=0.045", fill=False, ec=c, lw=1.7, zorder=5))


def icon_table(ax, cx, cy, s, c):
    rounded(ax, cx - s * 0.34, cy - s * 0.30, s * 0.68, s * 0.60, fc="none", ec=c, lw=1.8, r=0.004, z=5)
    for t in [-0.10, 0.10]:
        ax.plot([cx - s * 0.34, cx + s * 0.34], [cy + s * t, cy + s * t], color=c, lw=1.2, zorder=5)
    for t in [-0.11, 0.11]:
        ax.plot([cx + s * t, cx + s * t], [cy - s * 0.30, cy + s * 0.30], color=c, lw=1.2, zorder=5)


def icon_best(ax, cx, cy, s, c):
    ax.add_patch(Rectangle((cx - s * 0.30, cy - s * 0.24), s * 0.60, s * 0.48, fill=False, ec=c, lw=1.7, zorder=5))
    ax.text(cx, cy, "</>", ha="center", va="center", fontsize=11, fontweight="bold", color=c, zorder=6)
    ax.add_patch(Circle((cx - s * 0.22, cy + s * 0.16), s * 0.018, color=c, zorder=6))
    ax.add_patch(Circle((cx - s * 0.12, cy + s * 0.16), s * 0.018, color=c, zorder=6))


def icon_branch(ax, cx, cy, s, c):
    ax.plot([cx, cx], [cy + s * 0.28, cy - s * 0.05], color=c, lw=1.8, zorder=5)
    ax.plot([cx, cx - s * 0.26], [cy - s * 0.05, cy - s * 0.28], color=c, lw=1.8, zorder=5)
    ax.plot([cx, cx + s * 0.26], [cy - s * 0.05, cy - s * 0.28], color=c, lw=1.8, zorder=5)
    for px, py in [(cx, cy + s * 0.30), (cx, cy - s * 0.05), (cx - s * 0.28, cy - s * 0.30), (cx + s * 0.28, cy - s * 0.30)]:
        ax.add_patch(Circle((px, py), s * 0.055, fc="white", ec=c, lw=1.7, zorder=6))


def icon_sliders(ax, cx, cy, s, c):
    for xoff, knob in [(-0.20, 0.18), (0.0, -0.08), (0.20, 0.22)]:
        ax.plot([cx + s * xoff, cx + s * xoff], [cy - s * 0.30, cy + s * 0.30], color=c, lw=1.6, zorder=5)
        ax.add_patch(Circle((cx + s * xoff, cy + s * knob), s * 0.052, fc="white", ec=c, lw=1.7, zorder=6))


def icon_wrench(ax, cx, cy, s, c):
    ax.plot([cx - s * 0.24, cx + s * 0.22], [cy - s * 0.22, cy + s * 0.22], color=c, lw=2.0, zorder=5)
    ax.add_patch(Circle((cx - s * 0.30, cy - s * 0.28), s * 0.065, fill=False, ec=c, lw=1.7, zorder=6))
    ax.plot([cx + s * 0.18, cx + s * 0.32], [cy + s * 0.27, cy + s * 0.36], color=c, lw=1.8, zorder=5)


def row(ax, x, y, w, h, icon_fn, text, color):
    ax.plot([x + 0.03 * w, x + 0.97 * w], [y + h, y + h], color=COLORS["line"], lw=0.8, zorder=2)
    cx = x + 0.20 * w
    cy = y + h * 0.50
    icon_fn(ax, cx, cy, min(w, h) * 0.45, color)
    ax.text(
        x + 0.42 * w,
        cy,
        text,
        ha="left",
        va="center",
        fontsize=10.8,
        fontweight="bold",
        color=COLORS["ink"],
        linespacing=1.15,
        zorder=6,
    )


def tag(ax, x, y, text, color):
    rounded(ax, x, y, 0.115, 0.043, fc="white", ec=color, lw=1.0, r=0.006, z=9)
    ax.text(x + 0.0575, y + 0.0215, text, ha="center", va="center", fontsize=8.7, color=color, zorder=10)


def make():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 160,
            "savefig.dpi": 420,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(16.0, 8.4))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y, h, w, gap, x0 = 0.18, 0.70, 0.168, 0.032, 0.026
    xs = [x0 + i * (w + gap) for i in range(5)]
    headers = [
        ("1. TASK MODELING", COLORS["navy"]),
        ("2. REWARD PROPOSAL", COLORS["blue"]),
        ("3. TRAIN & EVAL", COLORS["teal"]),
        ("4. EVIDENCE SYSTEM", COLORS["amber"]),
        ("5. EVOLVE REWARD", COLORS["green"]),
    ]
    for x, (title, color) in zip(xs, headers):
        panel(ax, x, y, w, h, color, title)

    row_h = (h * 0.905) / 3
    top = y + h * 0.905
    content = [
        [(icon_env, "Anonymous\nEnvironment"), (icon_brain, "Env LLM\nSummary"), (icon_doc, "Environment\nCard")],
        [(icon_plan, "Reward\nProposal"), (icon_code, "Reward Code\nr_t"), (icon_shield, "Interface\nValidation")],
        [(icon_network, "PPO\nTraining"), (icon_bars, "External\nReturn"), (icon_search, "Component\nEvidence")],
        [(icon_db, "Reward\nMemory"), (icon_table, "Component\nStats Table"), (icon_best, "Previous +\nBest Code")],
        [(icon_branch, "L1 / L2 / L3\nPolicy"), (icon_sliders, "One-Component\nRepair"), (icon_wrench, "Reward Code\nr_{t+1}")],
    ]
    for x, (_, color), rows in zip(xs, headers, content):
        for i, item in enumerate(rows):
            row(ax, x, top - (i + 1) * row_h, w, row_h, item[0], item[1], color)

    for i in range(4):
        arrow(ax, xs[i] + w + 0.006, y + h * 0.50, xs[i + 1] - 0.006, y + h * 0.50, color=COLORS["black"])

    # Iteration loop: from evolved reward back to reward proposal.
    loop_y = 0.085
    ax.plot([xs[4] + w * 0.50, xs[4] + w * 0.50, xs[1] + w * 0.50, xs[1] + w * 0.50], [y, loop_y, loop_y, y], color=COLORS["black"], lw=2.0, zorder=1)
    arrow(ax, xs[1] + w * 0.50, y - 0.006, xs[1] + w * 0.50, y + 0.010, color=COLORS["black"], lw=2.0, ms=15)
    ax.text(0.50, loop_y + 0.028, "NEXT ITERATION", ha="center", va="center", fontsize=13.2, fontweight="bold", color=COLORS["black"])

    # Ablation anchors, kept small and local.
    tag(ax, xs[0] + 0.026, y + h + 0.018, "w/o Env LLM", COLORS["navy"])
    tag(ax, xs[3] + 0.026, y + h + 0.018, "w/o Evidence", COLORS["amber"])
    tag(ax, xs[4] + 0.026, y + h + 0.018, "w/o L1/L2/L3", COLORS["green"])

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig02_deres_agent_framework_refined.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    make()
