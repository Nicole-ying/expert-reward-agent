"""Regenerate the two vector figures used by the ISCSIC paper.
Run from the paper directory: python scripts_generate_figures.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Framework diagram
fig, ax = plt.subplots(figsize=(7.15, 1.55))
ax.set_xlim(0, 12); ax.set_ylim(0, 3.2); ax.axis("off")
boxes = [
    (0.15,1.45,1.65,0.75,"Task interface"),
    (2.05,1.45,1.55,0.75,"Reward program\n$R_t$"),
    (3.90,1.45,1.55,0.75,"PPO policy\ntraining"),
    (5.75,1.45,1.80,0.75,"Native-task\nevaluation"),
    (7.85,1.45,1.75,0.75,"Training evidence\nsummary"),
    (9.90,1.45,1.85,0.75,"Reward revision\nagent"),
]
for x,y,w,h,text in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.07",facecolor="white",edgecolor="black",linewidth=1))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=8)
for left,right in zip(boxes[:-1],boxes[1:]):
    x,y,w,h,_=left; nx,ny,nw,nh,_=right
    ax.add_patch(FancyArrowPatch((x+w,y+h/2),(nx,ny+nh/2),arrowstyle="-|>",mutation_scale=9,linewidth=1,color="black"))
ax.add_patch(FancyArrowPatch((10.8,1.42),(2.8,1.42),connectionstyle="arc3,rad=-0.30",arrowstyle="-|>",mutation_scale=9,linewidth=1,color="black"))
ax.text(6.8,0.32,"Next reward $R_{t+1}$",ha="center",va="center",fontsize=8)
for x,text in [(4.35,"Lineage memory"),(7.0,"Best archive")]:
    ax.add_patch(FancyBboxPatch((x,2.55),1.75,0.48,boxstyle="round,pad=0.03,rounding_size=0.05",facecolor="0.95",edgecolor="black",linewidth=.9))
    ax.text(x+.875,2.79,text,ha="center",va="center",fontsize=7.5)
ax.add_patch(FancyArrowPatch((5.22,2.55),(8.7,2.22),arrowstyle="-|>",mutation_scale=8,linewidth=.8,color="black"))
ax.add_patch(FancyArrowPatch((7.88,2.55),(6.65,2.22),arrowstyle="-|>",mutation_scale=8,linewidth=.8,color="black"))
ax.text(4.67,1.18,"generated reward only",ha="center",va="center",fontsize=6.8)
ax.text(6.65,1.18,"native reward only",ha="center",va="center",fontsize=6.8)
fig.savefig(FIG/"create_framework.pdf",bbox_inches="tight",pad_inches=0.02)
plt.close(fig)

# Multi-seed curves
df = pd.read_csv(ROOT/"data"/"iteration_scores.csv")
fig, axes = plt.subplots(1,2,figsize=(7.15,2.35),constrained_layout=True)
for ax,(env,threshold,title) in zip(axes,[("LunarLander-v3",200,"LunarLander-v3"),("BipedalWalker-v3",300,"BipedalWalker-v3")]):
    part=df[df.environment==env]
    for seed,g in part.groupby("seed"):
        g=g.sort_values("iteration")
        ax.plot(g.iteration,g.score,marker="o",markersize=2.7,linewidth=1,label=f"Seed {seed}")
    ax.axhline(threshold,linestyle="--",linewidth=1,color="black",label="Threshold")
    ax.set_title(title,fontsize=9); ax.set_xlabel("Reward evaluation",fontsize=8); ax.set_ylabel("Native task return",fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(True,alpha=.25,linewidth=.5); ax.set_xticks(range(1,int(part.iteration.max())+1))
axes[0].legend(fontsize=6,ncol=2,loc="lower left",frameon=True)
axes[1].legend(fontsize=6,ncol=2,loc="lower right",frameon=True)
fig.savefig(FIG/"reward_evolution_curves.pdf",bbox_inches="tight")
plt.close(fig)
