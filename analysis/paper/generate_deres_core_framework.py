from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "paper" / "deres_main"

W, H = 2200, 1220

COLORS = {
    "ink": "#101827",
    "muted": "#566579",
    "line": "#D8DEE7",
    "bg": "#FBFCFE",
    "panel": "#FFFFFF",
    "shadow": "#D6DEE8",
    "blue": "#093A64",
    "blue2": "#0A5DB8",
    "teal": "#087A78",
    "orange": "#B97700",
    "green": "#1E7B25",
    "gray": "#475569",
}


def rounded(ax, x, y, w, h, fc, ec, lw=2.4, radius=18, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, s, size=23, color=None, weight="bold", ha="left", va="center", z=20, linespacing=1.05):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=color or COLORS["ink"],
        fontweight=weight,
        ha=ha,
        va=va,
        zorder=z,
        linespacing=linespacing,
        family="DejaVu Sans",
    )


def line(ax, xs, ys, color, lw=3, z=10):
    ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle="round", zorder=z)


def arrow(ax, start, end, color="#222222", lw=4, ms=24, rad=0.0, z=12):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
            zorder=z,
        )
    )


def icon_document(ax, cx, cy, s, c):
    w, h = s * 0.55, s * 0.72
    x, y = cx - w / 2, cy - h / 2
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=c, linewidth=3.4, zorder=18))
    ax.add_patch(Polygon([[x + w * 0.68, y], [x + w, y + h * 0.22], [x + w * 0.68, y + h * 0.22]],
                         fill=False, edgecolor=c, linewidth=3.4, zorder=18))
    for yy in [0.43, 0.58, 0.73]:
        line(ax, [x + w * 0.18, x + w * 0.78], [y + h * yy, y + h * yy], c, lw=2.8, z=18)


def icon_network(ax, cx, cy, s, c):
    pts = [
        (cx, cy - s * 0.34),
        (cx - s * 0.32, cy - s * 0.08),
        (cx + s * 0.32, cy - s * 0.08),
        (cx - s * 0.23, cy + s * 0.30),
        (cx + s * 0.23, cy + s * 0.30),
    ]
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 4), (1, 2), (3, 4)]:
        line(ax, [pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], c, lw=3, z=18)
    for px, py in pts:
        ax.add_patch(Circle((px, py), s * 0.095, facecolor=COLORS["panel"], edgecolor=c, linewidth=3.2, zorder=19))


def icon_card(ax, cx, cy, s, c):
    rounded(ax, cx - s * 0.36, cy - s * 0.30, s * 0.72, s * 0.60, "#FFFFFF", c, lw=3.2, radius=7, z=18)
    for i in range(3):
        line(ax, [cx - s * 0.23, cx + s * 0.23], [cy - s * 0.10 + i * s * 0.14] * 2, c, lw=2.8, z=19)
    ax.add_patch(Circle((cx - s * 0.24, cy - s * 0.19), s * 0.035, facecolor=c, edgecolor=c, zorder=19))
    ax.add_patch(Circle((cx - s * 0.10, cy - s * 0.19), s * 0.035, facecolor=c, edgecolor=c, zorder=19))


def icon_prior(ax, cx, cy, s, c):
    pts = [(cx, cy - s * 0.32), (cx - s * 0.30, cy + s * 0.22), (cx + s * 0.30, cy + s * 0.22)]
    for px, py in pts[1:]:
        line(ax, [cx, px], [cy - s * 0.15, py], c, lw=3.2, z=18)
    line(ax, [pts[1][0], pts[2][0]], [pts[1][1], pts[2][1]], c, lw=3.2, z=18)
    for px, py in pts:
        ax.add_patch(Circle((px, py), s * 0.085, facecolor=COLORS["panel"], edgecolor=c, linewidth=3.2, zorder=19))


def icon_code(ax, cx, cy, s, c):
    rounded(ax, cx - s * 0.40, cy - s * 0.28, s * 0.80, s * 0.56, "#FFFFFF", c, lw=3.2, radius=7, z=18)
    for i in range(3):
        ax.add_patch(Circle((cx - s * 0.28 + i * s * 0.12, cy - s * 0.17), s * 0.032, facecolor=c, edgecolor=c, zorder=19))
    text(ax, cx - s * 0.20, cy + s * 0.09, "</>", size=17, color=c, weight="bold", ha="left", z=19)


def icon_shield(ax, cx, cy, s, c):
    pts = [
        (cx, cy - s * 0.38),
        (cx + s * 0.34, cy - s * 0.20),
        (cx + s * 0.26, cy + s * 0.20),
        (cx, cy + s * 0.40),
        (cx - s * 0.26, cy + s * 0.20),
        (cx - s * 0.34, cy - s * 0.20),
    ]
    ax.add_patch(Polygon(pts, fill=False, edgecolor=c, linewidth=3.4, zorder=18))
    line(ax, [cx - s * 0.15, cx - s * 0.02, cx + s * 0.20], [cy + s * 0.03, cy + s * 0.17, cy - s * 0.14], c, lw=4, z=19)


def icon_bars(ax, cx, cy, s, c):
    base = cy + s * 0.32
    widths = s * 0.10
    for i, h in enumerate([0.25, 0.42, 0.63]):
        x = cx - s * 0.23 + i * s * 0.20
        ax.add_patch(Rectangle((x, base - s * h), widths, s * h, facecolor=c, edgecolor=c, zorder=18))
    line(ax, [cx - s * 0.32, cx + s * 0.34], [base, base], c, lw=3.2, z=18)


def icon_table(ax, cx, cy, s, c):
    rounded(ax, cx - s * 0.33, cy - s * 0.30, s * 0.66, s * 0.60, "#FFFFFF", c, lw=3.2, radius=7, z=18)
    for dx in [-0.11, 0.11]:
        line(ax, [cx + s * dx, cx + s * dx], [cy - s * 0.30, cy + s * 0.30], c, lw=2.7, z=19)
    for dy in [-0.10, 0.10]:
        line(ax, [cx - s * 0.33, cx + s * 0.33], [cy + s * dy, cy + s * dy], c, lw=2.7, z=19)


def icon_search(ax, cx, cy, s, c):
    ax.add_patch(Circle((cx - s * 0.08, cy - s * 0.08), s * 0.25, fill=False, edgecolor=c, linewidth=3.4, zorder=18))
    line(ax, [cx + s * 0.10, cx + s * 0.34], [cy + s * 0.10, cy + s * 0.34], c, lw=4, z=18)
    ax.add_patch(Circle((cx + s * 0.20, cy - s * 0.22), s * 0.045, facecolor=c, edgecolor=c, zorder=19))


def icon_sliders(ax, cx, cy, s, c):
    xs = [cx - s * 0.22, cx, cx + s * 0.22]
    knobs = [cy + s * 0.14, cy - s * 0.16, cy + s * 0.02]
    for x, k in zip(xs, knobs):
        line(ax, [x, x], [cy - s * 0.32, cy + s * 0.32], c, lw=3, z=18)
        ax.add_patch(Circle((x, k), s * 0.075, facecolor=COLORS["panel"], edgecolor=c, linewidth=3, zorder=19))


def icon_repair(ax, cx, cy, s, c):
    ax.add_patch(Circle((cx - s * 0.18, cy + s * 0.17), s * 0.075, facecolor=COLORS["panel"], edgecolor=c, linewidth=3, zorder=19))
    ax.add_patch(Circle((cx + s * 0.19, cy - s * 0.20), s * 0.075, facecolor=COLORS["panel"], edgecolor=c, linewidth=3, zorder=19))
    line(ax, [cx - s * 0.12, cx + s * 0.12], [cy + s * 0.11, cy - s * 0.13], c, lw=5, z=18)
    line(ax, [cx + s * 0.08, cx + s * 0.29], [cy - s * 0.30, cy - s * 0.09], c, lw=3.2, z=18)


def icon_decision(ax, cx, cy, s, c):
    centers = [(cx - s * 0.28, cy + s * 0.13), (cx, cy + s * 0.13), (cx + s * 0.28, cy + s * 0.13)]
    labels = ["✓", "×", "↑"]
    colors = [COLORS["green"], "#CB1F29", c]
    for (px, py), lab, col in zip(centers, labels, colors):
        ax.add_patch(Circle((px, py), s * 0.115, facecolor="#FFFFFF", edgecolor=col, linewidth=3, zorder=18))
        text(ax, px, py - 2, lab, size=23, color=col, weight="bold", ha="center", va="center", z=19)
    line(ax, [cx, centers[0][0]], [cy - s * 0.25, centers[0][1] - s * 0.13], c, lw=2.8, z=18)
    line(ax, [cx, centers[1][0]], [cy - s * 0.25, centers[1][1] - s * 0.13], c, lw=2.8, z=18)
    line(ax, [cx, centers[2][0]], [cy - s * 0.25, centers[2][1] - s * 0.13], c, lw=2.8, z=18)
    ax.add_patch(Circle((cx, cy - s * 0.28), s * 0.095, facecolor="#FFFFFF", edgecolor=c, linewidth=3, zorder=19))


def icon_trophy(ax, cx, cy, s, c):
    rounded(ax, cx - s * 0.16, cy - s * 0.22, s * 0.32, s * 0.30, "#FFFFFF", c, lw=3, radius=6, z=18)
    ax.add_patch(Ellipse((cx - s * 0.23, cy - s * 0.08), s * 0.20, s * 0.22, fill=False, edgecolor=c, linewidth=3, zorder=18))
    ax.add_patch(Ellipse((cx + s * 0.23, cy - s * 0.08), s * 0.20, s * 0.22, fill=False, edgecolor=c, linewidth=3, zorder=18))
    line(ax, [cx, cx], [cy + s * 0.08, cy + s * 0.28], c, lw=3.5, z=18)
    line(ax, [cx - s * 0.18, cx + s * 0.18], [cy + s * 0.30, cy + s * 0.30], c, lw=3.5, z=18)
    text(ax, cx, cy - s * 0.06, "★", size=20, color=c, weight="bold", ha="center", va="center", z=19)


ICON = {
    "document": icon_document,
    "network": icon_network,
    "card": icon_card,
    "prior": icon_prior,
    "code": icon_code,
    "shield": icon_shield,
    "bars": icon_bars,
    "table": icon_table,
    "search": icon_search,
    "sliders": icon_sliders,
    "repair": icon_repair,
    "decision": icon_decision,
    "trophy": icon_trophy,
}


def draw_panel(ax, x, y, w, h, color, header, rows, core=False):
    rounded(ax, x + 8, y + 10, w, h, "#000000", "#000000", lw=0, radius=16, z=1).set_alpha(0.08)
    rounded(ax, x, y, w, h, COLORS["panel"], color, lw=3.2 if not core else 4.0, radius=16, z=2)
    rounded(ax, x, y, w, 92 if not core else 112, color, color, lw=0, radius=16, z=3)
    ax.add_patch(Rectangle((x, y + (74 if not core else 94)), w, 34, facecolor=color, edgecolor=color, linewidth=0, zorder=3))

    if core:
        text(ax, x + 24, y + 32, "4. CORE MODULE", size=14, color="#FFFFFF", weight="bold", z=20)
        text(ax, x + 24, y + 70, "REWARD SELF-EVOLUTION\nAGENT", size=12, color="#FFF2D7", weight="bold", z=20)
        row_top = y + 112
    else:
        text(ax, x + 24, y + 51, header, size=13.5, color="#FFFFFF", weight="bold", z=20)
        row_top = y + 92

    row_h = (h - (row_top - y)) / len(rows)
    for i, row in enumerate(rows):
        ry = row_top + i * row_h
        if i > 0:
            line(ax, [x, x + w], [ry, ry], COLORS["line"], lw=1.6, z=5)
        icon_x = x + 72
        icon_y = ry + row_h / 2
        ICON[row["icon"]](ax, icon_x, icon_y, 82 if not core else 72, color)
        text(ax, x + 138, icon_y - 17, row["title"], size=14.5 if not core else 12.5, color=COLORS["ink"], weight="bold", z=20)
        if row.get("sub"):
            text(ax, x + 138, icon_y + 19, row["sub"], size=10.0 if not core else 8.8, color=COLORS["muted"], weight="normal", z=20)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(22, 12.2), dpi=150)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    panel_y = 90
    panel_w = 380
    panel_h = 790
    gap = 54
    xs = [48 + i * (panel_w + gap) for i in range(5)]

    panels = [
        {
            "color": COLORS["blue"],
            "header": "1. TASK UNDERSTANDING",
            "rows": [
                {"icon": "document", "title": "Masked Env", "sub": "anonymous interface"},
                {"icon": "network", "title": "Env Analyzer LLM", "sub": "task semantics"},
                {"icon": "card", "title": "Env Card", "sub": "compact task model"},
            ],
        },
        {
            "color": COLORS["blue2"],
            "header": "2. REWARD PROPOSAL",
            "rows": [
                {"icon": "prior", "title": "Expert Priors", "sub": "roles, operators, risks"},
                {"icon": "code", "title": "Initial Reward", "sub": "reward code r_t"},
                {"icon": "shield", "title": "Validation", "sub": "executable reward"},
            ],
        },
        {
            "color": COLORS["teal"],
            "header": "3. TRAIN & EVALUATE",
            "rows": [
                {"icon": "network", "title": "Policy Training", "sub": "PPO on generated reward"},
                {"icon": "bars", "title": "External Return", "sub": "task score J_env"},
                {"icon": "table", "title": "Reward Evidence", "sub": "component traces"},
            ],
        },
        {
            "color": COLORS["orange"],
            "header": "",
            "core": True,
            "rows": [
                {"icon": "table", "title": "Evidence System", "sub": "score, code, memory"},
                {"icon": "search", "title": "Failure Diagnosis", "sub": "missing signal, scale, exploit"},
                {"icon": "sliders", "title": "Evolution Policy", "sub": "L1 scale, L2 form, L3 rebuild"},
                {"icon": "repair", "title": "Local Repair", "sub": "one targeted code change"},
            ],
        },
        {
            "color": COLORS["green"],
            "header": "5. ARCHIVE & ITERATE",
            "rows": [
                {"icon": "decision", "title": "Evolution Decision", "sub": "accept, revert, escalate"},
                {"icon": "code", "title": "Next Candidate", "sub": "reward code r_{t+1}"},
                {"icon": "trophy", "title": "Best Archive", "sub": "global best reward/policy"},
            ],
        },
    ]

    for i, panel in enumerate(panels):
        draw_panel(
            ax,
            xs[i],
            panel_y,
            panel_w,
            panel_h,
            panel["color"],
            panel["header"],
            panel["rows"],
            core=panel.get("core", False),
        )

    mid_y = panel_y + panel_h * 0.50
    for i in range(4):
        arrow(ax, (xs[i] + panel_w + 10, mid_y), (xs[i + 1] - 16, mid_y), color="#1B1B1B", lw=4.0, ms=30)

    loop_y = panel_y + panel_h + 115
    x5 = xs[4] + panel_w / 2
    x2 = xs[1] + panel_w / 2
    line(ax, [x5, x5], [panel_y + panel_h, loop_y], "#222222", lw=4.2, z=11)
    line(ax, [x5, x2], [loop_y, loop_y], "#222222", lw=4.2, z=11)
    arrow(ax, (x2, loop_y), (x2, panel_y + panel_h + 8), color="#222222", lw=4.2, ms=28)
    text(ax, (x2 + x5) / 2, loop_y - 34, "NEXT ITERATION", size=16, color=COLORS["ink"], weight="bold", ha="center", z=20)

    # Light emphasis ring around the scientific contribution.
    rounded(
        ax,
        xs[3] - 18,
        panel_y - 20,
        panel_w + 36,
        panel_h + 40,
        "none",
        "#E2A43A",
        lw=3.0,
        radius=24,
        z=1,
    ).set_alpha(0.85)

    out_png = OUT / "fig01_deres_core_framework.png"
    out_pdf = OUT / "fig01_deres_core_framework.pdf"
    out_svg = OUT / "fig01_deres_core_framework.svg"
    fig.savefig(out_png, dpi=240, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


if __name__ == "__main__":
    main()
