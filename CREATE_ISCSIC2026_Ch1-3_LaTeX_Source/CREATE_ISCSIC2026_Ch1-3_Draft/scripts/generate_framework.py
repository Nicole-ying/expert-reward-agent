from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.16, 2.40))
ax.set_xlim(0, 15.2)
ax.set_ylim(0, 5.0)
ax.axis("off")

blue, blue_fill = "#315EA8", "#EEF4FF"
teal, teal_fill = "#168C91", "#ECF9F8"
orange, orange_fill = "#D77A1F", "#FFF5E8"
purple, purple_fill = "#7253A6", "#F5F0FA"
gray, light_gray = "#5E6670", "#F5F6F7"


def box(x, y, w, h, text="", edge=gray, face="white", lw=1.0, fs=7.0, weight="normal", radius=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0.035,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face))
    if text:
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
                weight=weight, color="#1F2933", linespacing=1.08)


def arrow(x1, y1, x2, y2, color=gray, lw=1.0, rad=0.0, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
        mutation_scale=9, linewidth=lw, color=color, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5))

# Task interface
box(0.15, 1.38, 2.00, 2.75, edge=blue, lw=1.2, radius=0.12)
ax.text(1.15, 3.84, "1. Task Interface\n& Initialization", ha="center", va="center",
        fontsize=7.5, weight="bold", color=blue)
box(0.47, 2.70, 1.36, 0.62, "Task\nspecification", edge="#8997A5", face=light_gray, fs=6.7)
box(0.47, 1.78, 1.36, 0.62, "Initial reward\n$R_0$", edge="#8997A5", face=light_gray, fs=6.7)
arrow(1.15, 2.67, 1.15, 2.43, color=blue, lw=0.85)

# CREATE agent
box(2.65, 0.35, 6.85, 4.45, edge=blue, face=blue_fill, lw=1.35, radius=0.14)
ax.text(6.075, 4.54, "2. CREATE Reward-Engineering Agent", ha="center", va="center",
        fontsize=8.5, weight="bold", color=blue)

# Observe
box(2.98, 3.22, 6.18, 1.04, edge=blue, face="white", lw=0.9)
ax.text(3.18, 4.02, "Observe", ha="left", va="center", fontsize=7.3, weight="bold", color=blue)
for x, w, text in [
    (3.25, 1.10, "Outcome"),
    (4.48, 1.16, "Dynamics"),
    (5.77, 1.31, "Components"),
    (7.21, 1.12, "Termination"),
]:
    box(x, 3.43, w, 0.43, text, edge="#7892BE", face=blue_fill, fs=6.0)
box(8.46, 3.43, 0.47, 0.43, "$O_t$", edge=purple, face=purple_fill, fs=7.0, weight="bold")

# Reflect and plan
box(2.98, 1.91, 6.18, 1.04, edge=blue, face="white", lw=0.9)
ax.text(3.18, 2.71, "Reflect & Plan", ha="left", va="center", fontsize=7.3, weight="bold", color=blue)
box(3.25, 2.12, 1.55, 0.43, "Evidence tools", edge=purple, face=purple_fill, fs=5.9)
box(4.98, 2.12, 1.43, 0.43, "Reflection LLM", edge=purple, face=purple_fill, fs=5.9)
box(6.60, 2.12, 0.70, 0.43, "L1\nTune", edge="#7892BE", face=blue_fill, fs=5.9)
box(7.43, 2.12, 0.70, 0.43, "L2\nRefactor", edge="#7892BE", face=blue_fill, fs=5.9)
box(8.26, 2.12, 0.70, 0.43, "L3\nRedesign", edge="#D9A25D", face=orange_fill, fs=5.9)
arrow(4.82, 2.335, 4.96, 2.335, color=purple, lw=0.75)
arrow(6.43, 2.335, 6.58, 2.335, color=blue, lw=0.75)

# Act / validate / remember
box(2.98, 0.62, 6.18, 1.04, edge=blue, face="white", lw=0.9)
ax.text(3.18, 1.42, "Act, Validate & Remember", ha="left", va="center", fontsize=7.3, weight="bold", color=blue)
box(3.25, 0.83, 1.46, 0.39, "Reward edit", edge=blue, face=blue_fill, fs=5.9)
box(4.90, 0.83, 1.22, 0.39, "Validate", edge=blue, face=blue_fill, fs=5.9)
box(6.31, 0.83, 1.46, 0.39, "Memory", edge=purple, face=purple_fill, fs=5.9)
box(7.96, 0.83, 0.99, 0.39, "$R_{t+1}$", edge=blue, face=blue_fill, fs=6.8, weight="bold")
arrow(4.73, 1.025, 4.88, 1.025, color=blue, lw=0.75)
arrow(6.14, 1.025, 6.29, 1.025, color=blue, lw=0.75)
arrow(7.79, 1.025, 7.94, 1.025, color=blue, lw=0.75)
arrow(6.07, 3.19, 6.07, 2.98, color=blue, lw=0.9)
arrow(6.07, 1.89, 6.07, 1.68, color=blue, lw=0.9)

# Inner RL environment
box(10.20, 1.27, 2.55, 3.12, edge=teal, face=teal_fill, lw=1.3, radius=0.13)
ax.text(11.475, 4.08, "3. Inner RL Environment", ha="center", va="center",
        fontsize=8.1, weight="bold", color=teal)
box(10.56, 3.04, 1.83, 0.61, "Policy training\n(PPO in experiments)", edge=teal, face="white", fs=6.4)
box(10.56, 2.12, 1.83, 0.61, "Native-task\nevaluation", edge=teal, face="white", fs=6.4)
box(10.56, 1.48, 1.83, 0.36, r"Training record $\mathcal{D}_t$", edge=teal, face="white", fs=6.0)
arrow(11.475, 3.01, 11.475, 2.76, color=teal, lw=0.85)
arrow(11.475, 2.09, 11.475, 1.87, color=teal, lw=0.85)

# Best archive
box(13.34, 1.63, 1.62, 2.48, edge=orange, face=orange_fill, lw=1.2, radius=0.12)
ax.text(14.15, 3.82, "4. Best Archive", ha="center", va="center",
        fontsize=8.0, weight="bold", color=orange)
box(13.63, 2.72, 1.04, 0.58, "Guarded\nbest reward", edge=orange, face="white", fs=6.2)
box(13.63, 2.00, 1.04, 0.42, "Final $R^*$", edge=orange, face="white", fs=6.8, weight="bold")
arrow(14.15, 2.69, 14.15, 2.44, color=orange, lw=0.85)

# External flow, kept entirely in gaps and outside inner text boxes.
arrow(2.16, 2.06, 2.96, 2.06, color=blue, lw=1.1)
ax.text(2.56, 2.22, "$R_0$", ha="center", va="bottom", fontsize=6.5, color=blue)

# Reward action to policy training
arrow(9.18, 1.04, 10.54, 3.34, color=teal, lw=1.15, rad=-0.12)
ax.text(9.87, 2.04, "$R_t$ / $R_{t+1}$", ha="center", va="center", fontsize=6.1, color=teal)

# Training evidence back to observation via top arc
arrow(10.57, 3.55, 9.12, 3.82, color=blue, lw=1.15, rad=0.24)

# Native evaluation to archive
arrow(12.77, 2.42, 13.61, 2.94, color=orange, lw=1.0, rad=0.03)
ax.text(13.10, 2.78, "$J_t$", ha="center", va="bottom", fontsize=6.3, color=orange)

fig.savefig(OUT / "create_framework.pdf", bbox_inches="tight", pad_inches=0.025)
fig.savefig(OUT / "create_framework.png", dpi=300, bbox_inches="tight", pad_inches=0.025)
plt.close(fig)
