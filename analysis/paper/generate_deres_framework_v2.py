"""Generate the mechanism-level DERES overview used as Figure 1."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "paper" / "deres_main"
NAVY, BLUE, TEAL, ORANGE, GREEN = "#173B57", "#3579B8", "#188C86", "#D8872B", "#43865B"
INK, MUTED, PALE = "#17212B", "#5F6F7D", "#F5F7F8"


def box(ax, x, y, w, h, title, color, subtitle=None, lw=1.5, fill="white"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018,rounding_size=.08",
                       facecolor=fill, edgecolor=color, linewidth=lw)
    ax.add_patch(p)
    ax.text(x + .18, y + h - .28, title, color=color, fontsize=11, weight="bold", va="top")
    if subtitle:
        ax.text(x + .18, y + h - .62, subtitle, color=MUTED, fontsize=7.8, va="top", linespacing=1.25)
    return p


def arrow(ax, a, b, color=INK, rad=0, text=None, text_xy=None, lw=1.5, style="-|>"):
    p = FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, linewidth=lw,
                        color=color, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(p)
    if text and text_xy:
        ax.text(*text_xy, text, fontsize=7.8, color=color, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=.8))


def pill(ax, x, y, w, label, color, fill=None):
    fill = fill or color
    ax.add_patch(FancyBboxPatch((x, y), w, .34, boxstyle="round,pad=.03,rounding_size=.16",
                                facecolor=fill, edgecolor=color, linewidth=1))
    ax.text(x + w / 2, y + .17, label, ha="center", va="center", fontsize=7.3,
            color="white" if fill == color else color, weight="bold")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 7.4))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7.4); ax.axis("off")

    ax.text(.35, 7.05, "DERES: diagnosis-guided reward self-evolution", fontsize=16,
            weight="bold", color=INK)
    ax.text(.35, 6.72, "A single reward lineage is repaired from structured failure evidence — one attributable intervention at a time",
            fontsize=8.8, color=MUTED)

    # I. Semantic grounding
    ax.text(.35, 6.25, "I. SEMANTIC GROUNDING", fontsize=8.5, weight="bold", color=BLUE)
    box(ax, .35, 4.52, 2.45, 1.52, "Masked task interface", BLUE,
        "instruction  •  observation/action schema\nmasked transition code  •  signal constraints", fill="#F3F8FC")
    box(ax, 3.12, 4.52, 2.55, 1.52, "Reward-design specification", BLUE,
        "objective & failure conditions\nrole → usable signal → operator\nforbidden leakage checks", fill="#F3F8FC")
    arrow(ax, (2.80, 5.28), (3.12, 5.28), BLUE)
    ax.text(2.96, 5.48, "extract", fontsize=7, color=BLUE, ha="center")

    # II. Initial design
    ax.text(6.02, 6.25, "II. INITIAL DESIGN", fontsize=8.5, weight="bold", color=TEAL)
    box(ax, 6.02, 4.52, 2.55, 1.52, "Compact reward program  $R_1$", TEAL,
        "$R=\\sum_k w_k f_k(s,a,s')$\n2–4 interpretable components\nsyntax + signal-validity checks", fill="#F1FAF8")
    arrow(ax, (5.67, 5.28), (6.02, 5.28), TEAL)

    # III. execution
    ax.text(8.95, 6.25, "III. POLICY EXECUTION", fontsize=8.5, weight="bold", color=NAVY)
    box(ax, 8.95, 4.52, 2.45, 1.52, "Train policy  $\\pi_t$", NAVY,
        "PPO optimized with $R_t$\nfixed candidate budget $B$", fill="#F3F6F9")
    box(ax, 11.75, 4.52, 2.45, 1.52, "Evaluate behavior", NAVY,
        "external task return $J_{env}$\nepisode outcomes + component traces", fill="#F3F6F9")
    arrow(ax, (8.57, 5.28), (8.95, 5.28), NAVY)
    arrow(ax, (11.40, 5.28), (11.75, 5.28), NAVY)

    # Evidence bus
    ax.text(.35, 3.95, "IV. DIAGNOSTIC REPAIR", fontsize=8.5, weight="bold", color=ORANGE)
    box(ax, .35, 1.10, 4.10, 2.62, "Reward evidence dossier  $E_t$", ORANGE,
        "failure is retained as design evidence", fill="#FFF8ED")
    labels = [("Task outcome", "return • length • termination"),
              ("Component behavior", "magnitude share • active rate"),
              ("Design lineage", "$R_t$ ↔ $R^*$ • hypotheses • outcomes")]
    for i, (a, b) in enumerate(labels):
        yy = 2.65 - i * .62
        ax.add_patch(Circle((.72, yy + .05), .12, facecolor=ORANGE, edgecolor="none"))
        ax.text(.98, yy + .12, a, fontsize=8.5, weight="bold", color=INK, va="center")
        ax.text(.98, yy - .12, b, fontsize=7.3, color=MUTED, va="center")

    box(ax, 4.80, 1.10, 3.18, 2.62, "Failure hypothesis", ORANGE,
        "Identify the smallest plausible cause", fill="#FFF8ED")
    for i, s in enumerate(["inactive progress signal", "dominant safety penalty", "misaligned mathematical form"]):
        ax.text(5.08, 2.76 - i * .48, f"{i+1}.  {s}", fontsize=8, color=INK)
    ax.text(5.08, 1.40, "Select one target component  $k^*$", fontsize=8.4, color=ORANGE, weight="bold")

    box(ax, 8.34, 1.10, 3.48, 2.62, "Hierarchical intervention", ORANGE,
        "Escalate only when a lower-cost repair fails", fill="#FFF8ED")
    pill(ax, 8.65, 2.64, .76, "L1", ORANGE)
    ax.text(9.58, 2.81, "recalibrate", fontsize=8.4, weight="bold", color=INK)
    ax.text(9.58, 2.59, "weight / scale / threshold", fontsize=7.2, color=MUTED)
    pill(ax, 8.65, 2.02, .76, "L2", ORANGE)
    ax.text(9.58, 2.19, "transform", fontsize=8.4, weight="bold", color=INK)
    ax.text(9.58, 1.97, "sparse→dense • absolute→Δ", fontsize=7.2, color=MUTED)
    pill(ax, 8.65, 1.40, .76, "L3", ORANGE)
    ax.text(9.58, 1.57, "recompose", fontsize=8.4, weight="bold", color=INK)
    ax.text(9.58, 1.35, "rebuild reward skeleton", fontsize=7.2, color=MUTED)

    box(ax, 12.18, 1.10, 3.47, 2.62, "Attributable program edit", GREEN,
        "exactly one component changes at L1/L2", fill="#F2F9F4")
    ax.text(12.52, 2.67, "before", fontsize=7, color=MUTED, weight="bold")
    ax.text(13.78, 2.67, "after", fontsize=7, color=MUTED, weight="bold")
    ax.text(12.52, 2.28, "$0.1\\,r_{progress}$", fontsize=9, color="#A34A43", family="monospace")
    ax.text(13.78, 2.28, "$0.8\\,r_{progress}$", fontsize=9, color=GREEN, family="monospace")
    arrow(ax, (13.47, 2.34), (13.72, 2.34), GREEN)
    ax.text(12.52, 1.82, "$R_{t+1}=R_t-r_{k^*}+\\tilde r_{k^*}$", fontsize=9.2, color=INK)
    ax.text(12.52, 1.42, "accept • revert • escalate  →  best archive", fontsize=7.5, color=MUTED)

    arrow(ax, (13.25, 4.52), (2.40, 3.72), ORANGE, rad=.18,
          text="evaluation outcomes", text_xy=(7.70, 4.08))
    arrow(ax, (4.45, 2.42), (4.80, 2.42), ORANGE)
    arrow(ax, (7.98, 2.42), (8.34, 2.42), ORANGE)
    arrow(ax, (11.82, 2.42), (12.18, 2.42), ORANGE)

    # Loop-back: repaired program enters next policy execution.
    arrow(ax, (15.15, 1.10), (10.20, 4.52), GREEN, rad=-.28,
          text="$R_{t+1}$ / next iteration", text_xy=(13.52, .62), lw=2.0)
    ax.text(.35, .35, "Key distinction", fontsize=7.4, color=ORANGE, weight="bold")
    ax.text(1.37, .35, "DERES searches in depth along one lineage; discarded failures become structured evidence for the next repair.",
            fontsize=7.4, color=MUTED)

    fig.tight_layout(pad=.25)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig01_deres_mechanism_framework.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
