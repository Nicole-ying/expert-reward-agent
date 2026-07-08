from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "blue": "#276FBF",
    "blue_dark": "#174A7C",
    "green": "#2E8B57",
    "red": "#B84A4A",
    "amber": "#D4922E",
    "gray": "#9AA5B1",
    "gray_dark": "#4F5B66",
    "grid": "#E4E9EF",
    "text": "#20262E",
}


def apply_paper_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.edgecolor": "#BCC5CE",
            "axes.linewidth": 0.7,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.55,
            "grid.alpha": 0.9,
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3)
    ax.set_axisbelow(True)


def save_figure(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

