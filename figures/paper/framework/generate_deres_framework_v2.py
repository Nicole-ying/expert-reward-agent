from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT = Path(__file__).resolve().parent


COLORS = {
    "navy": "#0E355C",
    "blue": "#145EA8",
    "teal": "#087C7B",
    "amber": "#B87405",
    "green": "#237A1E",
    "ink": "#111827",
    "muted": "#4B5563",
    "line": "#C7CED6",
    "panel": "#FBFCFE",
    "soft": "#F7FAFC",
    "red": "#CC2F2F",
}


def rounded(ax, xy, w, h, fc="white", ec="#222", lw=1.4, r=0.025, z=2):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def panel(ax, x, y, w, h, color, title):
    rounded(ax, (x, y), w, h, fc=COLORS["panel"], ec=color, lw=1.6, r=0.018)
    header_h = 0.12 * h
    ax.add_patch(
        Rectangle((x, y + h - header_h), w, header_h, facecolor=color, edgecolor=color, zorder=3)
    )
    ax.text(
        x + 0.018 * w,
        y + h - header_h / 2,
        title,
        ha="left",
        va="center",
        fontsize=10.8,
        fontweight="bold",
        color="white",
        zorder=4,
    )
    return header_h


def _abbr(text: str) -> str:
    first = text.split("\n", 1)[0].strip()
    mapping = {
        "Masked": "ENV",
        "Environment": "LLM" if "LLM" in text else "CARD",
        "Expert": "RAG",
        "Reward": "CODE",
        "Interface": "VAL",
        "PPO": "PPO",
        "External": "RET",
        "Component": "STAT",
        "Previous": "BEST",
        "L1": "L1-3",
        "One-Component": "ONE",
    }
    for key, value in mapping.items():
        if first.startswith(key):
            return value
    return first[:3].upper()


def row(ax, x, y, w, h, icon_fn, text, color):
    ax.plot([x, x + w], [y + h, y + h], color=COLORS["line"], lw=0.8, zorder=4)
    icon_size = min(0.30 * w, 0.55 * h)
    bx = x + 0.075 * w
    by = y + 0.50 * h - icon_size / 2
    rounded(ax, (bx, by), icon_size, icon_size, fc="white", ec=color, lw=1.6, r=0.008, z=5)
    ax.text(
        bx + icon_size / 2,
        by + icon_size / 2,
        _abbr(text),
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=color,
        zorder=6,
    )
    ax.text(
        x + 0.43 * w,
        y + 0.50 * h,
        text,
        ha="left",
        va="center",
        fontsize=9.7,
        fontweight="bold",
        color=COLORS["ink"],
        linespacing=1.25,
        zorder=5,
    )


def small_tag(ax, x, y, text, color):
    rounded(ax, (x, y), 0.118, 0.035, fc="white", ec=color, lw=1.0, r=0.008, z=6)
    ax.text(x + 0.059, y + 0.0175, text, ha="center", va="center", fontsize=8.5, color=color, zorder=7)


def icon_globe(ax, x, y, s, c):
    ax.add_patch(Circle((x + s / 2, y + s / 2), s * 0.42, fill=False, ec=c, lw=1.8, zorder=5))
    ax.plot([x + s * 0.08, x + s * 0.92], [y + s * 0.50, y + s * 0.50], color=c, lw=1.5, zorder=5)
    ax.plot([x + s * 0.50, x + s * 0.50], [y + s * 0.10, y + s * 0.90], color=c, lw=1.5, zorder=5)
    ax.add_patch(FancyBboxPatch((x + s * 0.24, y + s * 0.10), s * 0.52, s * 0.80, boxstyle="round,pad=0,rounding_size=0.08", fill=False, ec=c, lw=1.2, zorder=5))
    ax.add_patch(FancyBboxPatch((x + s * 0.10, y + s * 0.25), s * 0.80, s * 0.50, boxstyle="round,pad=0,rounding_size=0.08", fill=False, ec=c, lw=1.2, zorder=5))


def icon_card(ax, x, y, s, c):
    ax.add_patch(Rectangle((x + s * 0.15, y + s * 0.12), s * 0.58, s * 0.72, fill=False, ec=c, lw=1.8, zorder=5))
    ax.add_patch(Polygon([[x + s * 0.73, y + s * 0.84], [x + s * 0.90, y + s * 0.67], [x + s * 0.73, y + s * 0.67]], fill=False, ec=c, lw=1.8, zorder=5))
    for i in range(3):
        ax.plot([x + s * 0.27, x + s * 0.62], [y + s * (0.58 - i * 0.15), y + s * (0.58 - i * 0.15)], color=c, lw=1.4, zorder=5)


def icon_brain(ax, x, y, s, c):
    ax.add_patch(Circle((x + s * 0.36, y + s * 0.58), s * 0.19, fill=False, ec=c, lw=1.8, zorder=5))
    ax.add_patch(Circle((x + s * 0.58, y + s * 0.62), s * 0.22, fill=False, ec=c, lw=1.8, zorder=5))
    ax.add_patch(Circle((x + s * 0.50, y + s * 0.38), s * 0.22, fill=False, ec=c, lw=1.8, zorder=5))
    ax.plot([x + s * 0.38, x + s * 0.70], [y + s * 0.30, y + s * 0.18], color=c, lw=1.6, zorder=5)


def icon_code(ax, x, y, s, c):
    rounded(ax, (x + s * 0.08, y + s * 0.17), s * 0.82, s * 0.66, fc="none", ec=c, lw=1.8, r=0.006, z=5)
    ax.text(x + s * 0.32, y + s * 0.50, "</>", ha="center", va="center", fontsize=16, fontweight="bold", color=c, zorder=5)
    for i in range(3):
        ax.add_patch(Circle((x + s * (0.18 + 0.12 * i), y + s * 0.72), s * 0.025, color=c, zorder=5))


def icon_shield(ax, x, y, s, c):
    pts = [
        (x + s * 0.50, y + s * 0.88),
        (x + s * 0.86, y + s * 0.72),
        (x + s * 0.78, y + s * 0.30),
        (x + s * 0.50, y + s * 0.12),
        (x + s * 0.22, y + s * 0.30),
        (x + s * 0.14, y + s * 0.72),
    ]
    ax.add_patch(Polygon(pts, fill=False, ec=c, lw=1.8, zorder=5))
    ax.plot([x + s * 0.34, x + s * 0.47, x + s * 0.70], [y + s * 0.50, y + s * 0.36, y + s * 0.62], color=c, lw=1.8, zorder=5)


def icon_network(ax, x, y, s, c):
    pts = [(0.2, 0.28), (0.5, 0.78), (0.82, 0.30), (0.50, 0.42)]
    coords = [(x + s * a, y + s * b) for a, b in pts]
    for a, b in [(0, 1), (1, 2), (0, 3), (2, 3), (1, 3)]:
        ax.plot([coords[a][0], coords[b][0]], [coords[a][1], coords[b][1]], color=c, lw=1.5, zorder=5)
    for px, py in coords:
        ax.add_patch(Circle((px, py), s * 0.07, fc="white", ec=c, lw=1.8, zorder=5))


def icon_bars(ax, x, y, s, c):
    for i, h in enumerate([0.28, 0.48, 0.70]):
        ax.add_patch(Rectangle((x + s * (0.20 + i * 0.20), y + s * 0.16), s * 0.11, s * h, facecolor=c, edgecolor=c, zorder=5))
    ax.plot([x + s * 0.14, x + s * 0.78], [y + s * 0.16, y + s * 0.16], color=c, lw=1.6, zorder=5)


def icon_magnifier(ax, x, y, s, c):
    ax.add_patch(Circle((x + s * 0.42, y + s * 0.58), s * 0.22, fill=False, ec=c, lw=1.9, zorder=5))
    ax.plot([x + s * 0.58, x + s * 0.80], [y + s * 0.38, y + s * 0.16], color=c, lw=2.0, zorder=5)


def icon_database(ax, x, y, s, c):
    ax.add_patch(FancyBboxPatch((x + s * 0.18, y + s * 0.60), s * 0.64, s * 0.20, boxstyle="round,pad=0,rounding_size=0.06", fill=False, ec=c, lw=1.8, zorder=5))
    ax.add_patch(Rectangle((x + s * 0.18, y + s * 0.27), s * 0.64, s * 0.42, fill=False, ec=c, lw=1.8, zorder=5))
    ax.add_patch(FancyBboxPatch((x + s * 0.18, y + s * 0.17), s * 0.64, s * 0.20, boxstyle="round,pad=0,rounding_size=0.06", fill=False, ec=c, lw=1.8, zorder=5))


def icon_table(ax, x, y, s, c):
    rounded(ax, (x + s * 0.12, y + s * 0.16), s * 0.74, s * 0.66, fc="none", ec=c, lw=1.8, r=0.004, z=5)
    for k in [0.37, 0.58]:
        ax.plot([x + s * 0.12, x + s * 0.86], [y + s * k, y + s * k], color=c, lw=1.2, zorder=5)
    for k in [0.38, 0.62]:
        ax.plot([x + s * k, x + s * k], [y + s * 0.16, y + s * 0.82], color=c, lw=1.2, zorder=5)


def icon_branch(ax, x, y, s, c):
    ax.plot([x + s * 0.50, x + s * 0.50], [y + s * 0.76, y + s * 0.45], color=c, lw=1.8, zorder=5)
    ax.plot([x + s * 0.50, x + s * 0.24], [y + s * 0.45, y + s * 0.22], color=c, lw=1.8, zorder=5)
    ax.plot([x + s * 0.50, x + s * 0.50], [y + s * 0.45, y + s * 0.20], color=c, lw=1.8, zorder=5)
    ax.plot([x + s * 0.50, x + s * 0.76], [y + s * 0.45, y + s * 0.22], color=c, lw=1.8, zorder=5)
    for px, py in [(0.50, 0.78), (0.24, 0.20), (0.50, 0.18), (0.76, 0.20)]:
        ax.add_patch(Circle((x + s * px, y + s * py), s * 0.06, fc="white", ec=c, lw=1.8, zorder=5))


def icon_tune(ax, x, y, s, c):
    for px, knob in [(0.30, 0.60), (0.50, 0.35), (0.70, 0.68)]:
        ax.plot([x + s * px, x + s * px], [y + s * 0.18, y + s * 0.82], color=c, lw=1.6, zorder=5)
        ax.add_patch(Circle((x + s * px, y + s * knob), s * 0.055, fc="white", ec=c, lw=1.6, zorder=5))


def icon_wrench(ax, x, y, s, c):
    ax.plot([x + s * 0.26, x + s * 0.70], [y + s * 0.24, y + s * 0.68], color=c, lw=2.2, zorder=5)
    ax.add_patch(Circle((x + s * 0.22, y + s * 0.20), s * 0.07, fill=False, ec=c, lw=1.8, zorder=5))
    ax.plot([x + s * 0.66, x + s * 0.78], [y + s * 0.72, y + s * 0.84], color=c, lw=1.8, zorder=5)
    ax.plot([x + s * 0.72, x + s * 0.84], [y + s * 0.66, y + s * 0.78], color=c, lw=1.8, zorder=5)


def arrow(ax, x1, y1, x2, y2, color="#1F2937", lw=2.2, rad=0.0, dashed=False, ms=18):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=ms,
            color=color,
            linewidth=lw,
            connectionstyle=f"arc3,rad={rad}",
            linestyle=(0, (5, 4)) if dashed else "solid",
            zorder=8,
        )
    )


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
    fig, ax = plt.subplots(figsize=(16.0, 7.6))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x0, y0, w, h, gap = 0.022, 0.235, 0.174, 0.64, 0.026
    xs = [x0 + i * (w + gap) for i in range(5)]
    titles = [
        "1. ENV LLM",
        "2. REWARD SYNTHESIS",
        "3. TRAIN + EVIDENCE",
        "4. EVIDENCE SYSTEM",
        "5. EVOLUTION POLICY",
    ]
    cols = [COLORS["navy"], COLORS["blue"], COLORS["teal"], COLORS["amber"], COLORS["green"]]
    for x, title, col in zip(xs, titles, cols):
        panel(ax, x, y0, w, h, col, title)

    rh = (h * 0.88) / 3
    top = y0 + h * 0.88

    rows = [
        [(icon_globe, "Masked\nEnvironment"), (icon_brain, "Environment LLM\nTask Summary"), (icon_card, "Environment\nCard")],
        [(icon_database, "Expert Prior\nContext"), (icon_code, "Reward Code\nr_t"), (icon_shield, "Interface\nValidation")],
        [(icon_network, "PPO\nTraining"), (icon_bars, "External\nReturn"), (icon_magnifier, "Component\nEvidence")],
        [(icon_database, "Reward\nMemory"), (icon_table, "Component\nStats Table"), (icon_code, "Previous +\nBest Code")],
        [(icon_branch, "L1 / L2 / L3\nPolicy"), (icon_tune, "One-Component\nRepair"), (icon_wrench, "Reward Code\nr_{t+1}")],
    ]
    for pi, x in enumerate(xs):
        for ri, (ic, txt) in enumerate(rows[pi]):
            row(ax, x, top - (ri + 1) * rh, w, rh, ic, txt, cols[pi])

    # Ablation labels are close to modules, not explanatory footnotes.
    small_tag(ax, xs[0] + 0.023, y0 + h + 0.018, "w/o Env LLM", COLORS["navy"])
    small_tag(ax, xs[3] + 0.023, y0 + h + 0.018, "w/o Evidence", COLORS["amber"])
    small_tag(ax, xs[4] + 0.020, y0 + h + 0.018, "w/o L1/L2/L3", COLORS["green"])

    for i in range(4):
        arrow(ax, xs[i] + w + 0.006, y0 + 0.315, xs[i + 1] - 0.006, y0 + 0.315)

    # Iteration loop.
    loop_y = 0.105
    ax.plot([xs[4] + w * 0.50, xs[4] + w * 0.50, xs[1] + w * 0.50, xs[1] + w * 0.50], [y0, loop_y, loop_y, y0], color="#202020", lw=2.0, zorder=1)
    arrow(ax, xs[1] + w * 0.50, y0 - 0.004, xs[1] + w * 0.50, y0 + 0.001, color="#202020", lw=2.0, ms=15)
    ax.text(0.50, loop_y + 0.025, "NEXT ITERATION", ha="center", va="center", fontsize=13, fontweight="bold", color="#202020")

    # Minimal method title, no bottom slogan.
    ax.text(0.50, 0.955, "DERES: Diagnosis-guided Reward Function Self-Evolution", ha="center", va="center", fontsize=17, fontweight="bold", color=COLORS["ink"])

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig01_deres_agent_framework_v2.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    make()
