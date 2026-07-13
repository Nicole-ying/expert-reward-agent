from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "paper" / "deres_main"

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


def arrow(ax, x1, y1, x2, y2, color="#222", lw=2.35, ms=22, rad=0.0, dashed=False, z=10):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            linestyle=(0, (5, 3)) if dashed else "solid",
            zorder=z,
        )
    )


def panel(ax, x, y, w, h, color, title):
    rounded(ax, x, y, w, h, fc=COLORS["panel"], ec=color, lw=1.65, r=0.018, z=1)
    header_h = 0.125 * h
    ax.add_patch(Rectangle((x, y + h - header_h), w, header_h, facecolor=color, edgecolor=color, zorder=2))
    title_fs = 10.1 if "\n" in title else 10.6
    ax.text(
        x + 0.055 * w,
        y + h - header_h / 2,
        title,
        ha="left",
        va="center",
        fontsize=title_fs,
        fontweight="bold",
        color="white",
        linespacing=0.92,
        zorder=3,
    )
    return header_h


def icon_env(ax, cx, cy, s, color):
    ax.add_patch(Circle((cx, cy), s * 0.38, fill=False, ec=color, lw=1.8, zorder=5))
    ax.plot([cx - s * 0.35, cx + s * 0.35], [cy, cy], color=color, lw=1.35, zorder=5)
    ax.plot([cx, cx], [cy - s * 0.35, cy + s * 0.35], color=color, lw=1.35, zorder=5)
    ax.add_patch(FancyBboxPatch((cx - s * 0.20, cy - s * 0.36), s * 0.40, s * 0.72, boxstyle="round,pad=0,rounding_size=0.045", fill=False, ec=color, lw=1.15, zorder=5))


def icon_brain(ax, cx, cy, s, color):
    for dx, dy, rr in [(-0.16, 0.10, 0.15), (0.05, 0.16, 0.18), (0.18, -0.05, 0.14), (-0.04, -0.10, 0.20)]:
        ax.add_patch(Circle((cx + s * dx, cy + s * dy), s * rr, fill=False, ec=color, lw=1.75, zorder=5))
    ax.plot([cx - s * 0.10, cx + s * 0.25], [cy - s * 0.26, cy - s * 0.33], color=color, lw=1.55, zorder=5)


def icon_doc(ax, cx, cy, s, color):
    ax.add_patch(Rectangle((cx - s * 0.27, cy - s * 0.33), s * 0.48, s * 0.66, fill=False, ec=color, lw=1.85, zorder=5))
    ax.add_patch(Polygon([[cx + s * 0.21, cy + s * 0.33], [cx + s * 0.34, cy + s * 0.20], [cx + s * 0.21, cy + s * 0.20]], fill=False, ec=color, lw=1.8, zorder=5))
    for i in range(3):
        ax.plot([cx - s * 0.17, cx + s * 0.12], [cy + s * (0.08 - 0.14 * i), cy + s * (0.08 - 0.14 * i)], color=color, lw=1.25, zorder=5)


def icon_priors(ax, cx, cy, s, color):
    pts = [(cx, cy + s * 0.30), (cx - s * 0.28, cy - s * 0.16), (cx + s * 0.28, cy - s * 0.16), (cx, cy - s * 0.02)]
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)]:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=color, lw=1.35, zorder=5)
    for px, py in pts:
        ax.add_patch(Circle((px, py), s * 0.060, fc="white", ec=color, lw=1.65, zorder=6))


def icon_code(ax, cx, cy, s, color):
    rounded(ax, cx - s * 0.39, cy - s * 0.30, s * 0.78, s * 0.60, fc="none", ec=color, lw=1.9, r=0.004, z=5)
    ax.text(cx, cy, "</>", ha="center", va="center", fontsize=13.2, fontweight="bold", color=color, zorder=6)
    for i in range(3):
        ax.add_patch(Circle((cx - s * 0.28 + i * s * 0.09, cy + s * 0.20), s * 0.018, color=color, zorder=6))


def icon_shield(ax, cx, cy, s, color):
    pts = [(cx, cy + s * 0.38), (cx + s * 0.34, cy + s * 0.24), (cx + s * 0.26, cy - s * 0.20), (cx, cy - s * 0.38), (cx - s * 0.26, cy - s * 0.20), (cx - s * 0.34, cy + s * 0.24)]
    ax.add_patch(Polygon(pts, fill=False, ec=color, lw=1.85, zorder=5))
    ax.plot([cx - s * 0.15, cx - s * 0.02, cx + s * 0.22], [cy, cy - s * 0.13, cy + s * 0.12], color=color, lw=1.85, zorder=5)


def icon_network(ax, cx, cy, s, color):
    pts = [(cx - s * 0.28, cy - s * 0.18), (cx, cy + s * 0.32), (cx + s * 0.30, cy - s * 0.14), (cx, cy - s * 0.02)]
    for a, b in [(0, 1), (1, 2), (0, 3), (3, 2), (1, 3)]:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=color, lw=1.35, zorder=5)
    for px, py in pts:
        ax.add_patch(Circle((px, py), s * 0.065, fc="white", ec=color, lw=1.7, zorder=6))


def icon_bars(ax, cx, cy, s, color):
    for i, height in enumerate([0.25, 0.42, 0.62]):
        ax.add_patch(Rectangle((cx - s * 0.28 + i * s * 0.20, cy - s * 0.30), s * 0.10, s * height, facecolor=color, edgecolor=color, zorder=5))
    ax.plot([cx - s * 0.34, cx + s * 0.34], [cy - s * 0.30, cy - s * 0.30], color=color, lw=1.5, zorder=5)


def icon_search(ax, cx, cy, s, color):
    ax.add_patch(Circle((cx - s * 0.08, cy + s * 0.08), s * 0.22, fill=False, ec=color, lw=1.9, zorder=5))
    ax.plot([cx + s * 0.08, cx + s * 0.32], [cy - s * 0.10, cy - s * 0.32], color=color, lw=2.05, zorder=5)


def icon_table(ax, cx, cy, s, color):
    rounded(ax, cx - s * 0.34, cy - s * 0.30, s * 0.68, s * 0.60, fc="none", ec=color, lw=1.8, r=0.004, z=5)
    for t in [-0.10, 0.10]:
        ax.plot([cx - s * 0.34, cx + s * 0.34], [cy + s * t, cy + s * t], color=color, lw=1.2, zorder=5)
    for t in [-0.11, 0.11]:
        ax.plot([cx + s * t, cx + s * t], [cy - s * 0.30, cy + s * 0.30], color=color, lw=1.2, zorder=5)


def icon_db(ax, cx, cy, s, color):
    ax.add_patch(FancyBboxPatch((cx - s * 0.32, cy + s * 0.12), s * 0.64, s * 0.18, boxstyle="round,pad=0,rounding_size=0.045", fill=False, ec=color, lw=1.7, zorder=5))
    ax.add_patch(Rectangle((cx - s * 0.32, cy - s * 0.20), s * 0.64, s * 0.40, fill=False, ec=color, lw=1.7, zorder=5))
    ax.add_patch(FancyBboxPatch((cx - s * 0.32, cy - s * 0.30), s * 0.64, s * 0.18, boxstyle="round,pad=0,rounding_size=0.045", fill=False, ec=color, lw=1.7, zorder=5))


def icon_branch(ax, cx, cy, s, color):
    ax.plot([cx, cx], [cy + s * 0.30, cy - s * 0.05], color=color, lw=1.85, zorder=5)
    ax.plot([cx, cx - s * 0.27], [cy - s * 0.05, cy - s * 0.30], color=color, lw=1.85, zorder=5)
    ax.plot([cx, cx + s * 0.27], [cy - s * 0.05, cy - s * 0.30], color=color, lw=1.85, zorder=5)
    for px, py in [(cx, cy + s * 0.30), (cx, cy - s * 0.05), (cx - s * 0.29, cy - s * 0.31), (cx + s * 0.29, cy - s * 0.31)]:
        ax.add_patch(Circle((px, py), s * 0.055, fc="white", ec=color, lw=1.7, zorder=6))


def icon_sliders(ax, cx, cy, s, color):
    for xoff, knob in [(-0.20, 0.18), (0.0, -0.08), (0.20, 0.22)]:
        ax.plot([cx + s * xoff, cx + s * xoff], [cy - s * 0.30, cy + s * 0.30], color=color, lw=1.6, zorder=5)
        ax.add_patch(Circle((cx + s * xoff, cy + s * knob), s * 0.052, fc="white", ec=color, lw=1.7, zorder=6))


def icon_wrench(ax, cx, cy, s, color):
    ax.plot([cx - s * 0.24, cx + s * 0.22], [cy - s * 0.22, cy + s * 0.22], color=color, lw=2.0, zorder=5)
    ax.add_patch(Circle((cx - s * 0.30, cy - s * 0.28), s * 0.065, fill=False, ec=color, lw=1.7, zorder=6))
    ax.plot([cx + s * 0.18, cx + s * 0.32], [cy + s * 0.27, cy + s * 0.36], color=color, lw=1.8, zorder=5)


def icon_trophy(ax, cx, cy, s, color):
    ax.add_patch(Rectangle((cx - s * 0.16, cy - s * 0.30), s * 0.32, s * 0.10, fill=False, ec=color, lw=1.6, zorder=5))
    ax.add_patch(Rectangle((cx - s * 0.08, cy - s * 0.20), s * 0.16, s * 0.18, fill=False, ec=color, lw=1.6, zorder=5))
    ax.add_patch(FancyBboxPatch((cx - s * 0.22, cy - s * 0.02), s * 0.44, s * 0.34, boxstyle="round,pad=0,rounding_size=0.03", fill=False, ec=color, lw=1.8, zorder=5))
    ax.add_patch(Circle((cx - s * 0.29, cy + s * 0.13), s * 0.10, fill=False, ec=color, lw=1.4, zorder=5))
    ax.add_patch(Circle((cx + s * 0.29, cy + s * 0.13), s * 0.10, fill=False, ec=color, lw=1.4, zorder=5))
    ax.text(cx, cy + s * 0.12, "★", ha="center", va="center", fontsize=9.5, color=color, zorder=6)


def row(ax, x, y, w, h, icon_fn, text, color, first=False):
    if not first:
        ax.plot([x + 0.03 * w, x + 0.97 * w], [y + h, y + h], color=COLORS["line"], lw=0.8, zorder=2)
    cx = x + 0.20 * w
    cy = y + h * 0.50
    icon_fn(ax, cx, cy, min(w, h) * 0.43, color)
    ax.text(
        x + 0.40 * w,
        cy,
        text,
        ha="left",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=COLORS["ink"],
        linespacing=1.08,
        zorder=6,
    )


def generate():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 160,
            "savefig.dpi": 430,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(16.8, 7.6))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y, h, w, gap, x0 = 0.18, 0.72, 0.168, 0.032, 0.026
    xs = [x0 + i * (w + gap) for i in range(5)]
    headers = [
        ("1. TASK\nUNDERSTANDING", COLORS["navy"]),
        ("2. REWARD\nSYNTHESIS", COLORS["blue"]),
        ("3. TRAIN &\nEVALUATE", COLORS["teal"]),
        ("4. REWARD\nEVIDENCE", COLORS["amber"]),
        ("5. EVOLUTION\nAGENT", COLORS["green"]),
    ]
    for x, (title, color) in zip(xs, headers):
        panel(ax, x, y, w, h, color, title)

    content = [
        [
            (icon_env, "Anonymous\nEnvironment"),
            (icon_brain, "Env Analyzer\nLLM"),
            (icon_doc, "Environment\nCard"),
        ],
        [
            (icon_priors, "Expert Design\nPriors"),
            (icon_code, "Reward Code\n$R_t$"),
            (icon_shield, "Interface\nValidation"),
        ],
        [
            (icon_network, "PPO\nTraining"),
            (icon_code, "Generated\nReward"),
            (icon_bars, "External\nReturn"),
            (icon_search, "Component\nEvidence"),
        ],
        [
            (icon_table, "Score +\nOutcome"),
            (icon_table, "Component\nStatistics"),
            (icon_db, "Reward\nMemory"),
            (icon_doc, "Previous +\nBest Code"),
        ],
        [
            (icon_branch, "Failure\nDiagnosis"),
            (icon_sliders, "L1/L2/L3\nDecision"),
            (icon_wrench, "Local Reward\nRepair"),
            (icon_trophy, "Best\nArchive"),
        ],
    ]
    for x, (_, color), rows in zip(xs, headers, content):
        available_h = h * 0.895
        row_h = available_h / len(rows)
        top = y + available_h
        for i, item in enumerate(rows):
            row(ax, x, top - (i + 1) * row_h, w, row_h, item[0], item[1], color, first=i == 0)

    for i in range(4):
        arrow(ax, xs[i] + w + 0.006, y + h * 0.52, xs[i + 1] - 0.006, y + h * 0.52, color=COLORS["black"])

    # Compact evidence annotations between training and evidence.
    ax.text(xs[2] + w + 0.012, y + h * 0.62, "training\nfeedback", ha="left", va="center", fontsize=7.2, color=COLORS["teal"], fontweight="bold")
    ax.text(xs[2] + w + 0.012, y + h * 0.39, "diagnostic\nevidence", ha="left", va="center", fontsize=7.2, color=COLORS["teal"], fontweight="bold")

    # Iteration loop from evolution back to reward synthesis.
    loop_y = 0.085
    ax.plot(
        [xs[4] + w * 0.50, xs[4] + w * 0.50, xs[1] + w * 0.50, xs[1] + w * 0.50],
        [y, loop_y, loop_y, y],
        color=COLORS["black"],
        lw=2.1,
        zorder=1,
    )
    arrow(ax, xs[1] + w * 0.50, y - 0.006, xs[1] + w * 0.50, y + 0.012, color=COLORS["black"], lw=2.1, ms=15)
    ax.text(0.50, loop_y + 0.028, "NEXT REWARD CANDIDATE", ha="center", va="center", fontsize=12.8, fontweight="bold", color=COLORS["black"])

    OUT.mkdir(parents=True, exist_ok=True)
    stem = "fig01_deres_framework_main"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    generate()
