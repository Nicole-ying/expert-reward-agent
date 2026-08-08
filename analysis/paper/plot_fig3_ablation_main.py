"""Fig.3: Ablation scatter — adapted from plot_fig4_ablation.py"""
import matplotlib.pyplot as plt, matplotlib, numpy as np, json
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 8,
    'axes.labelsize': 9, 'axes.titlesize': 10,
    'legend.fontsize': 7, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

def load_best(base):
    scores = []
    for s in range(5):
        f = base / f'seed_{s}' / 'best' / 'best_training_summary.json'
        if f.exists():
            d = json.loads(f.read_text(encoding='utf-8'))
            scores.append(d['external_eval']['mean_eval_reward'])
    return scores

BASE = Path('/home/utseus22/expert-reward-agent/runs/env_001')
data = {
    'w/o Evidence':   load_best(BASE / 'ablation_eureka_feedback_v4'),
    'w/o Hierarchy':  load_best(BASE / 'ablation_unconstrained_v4'),
    'CREATE':         load_best(BASE / 'paper_v4'),
}

methods = list(data.keys())
n_methods = len(methods)
n_seeds = 5
THRESHOLD = 200

seed_colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd']
method_colors = ['#FF9800', '#9C27B0', '#2196F3']

fig, ax = plt.subplots(figsize=(3.45, 2.15))
rng = np.random.default_rng(42)

# Per-method scatter with jitter
for m, (label, scores) in enumerate(data.items()):
    xs = np.full(n_seeds, m) + rng.uniform(-0.12, 0.12, n_seeds)
    ax.scatter(xs, scores, c=seed_colors, s=65, zorder=3, edgecolors='white',
               linewidths=0.5, clip_on=False)

    # Mean bar
    mu = np.mean(scores)
    ax.plot([m - 0.25, m + 0.25], [mu, mu], '-', color=method_colors[m],
            linewidth=3.5, zorder=5, solid_capstyle='round')
    ax.annotate(f'{mu:.0f}', xy=(m, mu), xytext=(m, mu + 25 if mu > 0 else mu - 30),
                fontsize=7.5, ha='center', color=method_colors[m], fontweight='bold')

# Threshold
ax.axhline(y=THRESHOLD, color='#333', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
ax.annotate(f'Threshold={THRESHOLD}', xy=(2.35, THRESHOLD),
            fontsize=6.5, color='#333', ha='right', va='bottom')

ax.set_xticks(range(n_methods))
ax.set_xticklabels(methods)
ax.set_ylabel('Best search fitness')
ax.set_ylim(-155, 295)
ax.grid(axis='y', alpha=0.12, zorder=0)
for sp in ax.spines.values(): sp.set_linewidth(0.5)

# Seed legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                          markersize=7, label=f'seed_{i}')
                   for i, c in enumerate(seed_colors)]
ax.legend(handles=legend_elements, loc='lower right', ncol=5, framealpha=0.8,
          fontsize=6.5, columnspacing=0.5, handletextpad=0.3)

out_dir = Path(__file__).resolve().parent / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out_dir / 'fig3_ablation_scatter.pdf'))
fig.savefig(str(out_dir / 'fig3_ablation_scatter.png'), dpi=150)
print(f'Saved: {out_dir}/fig3_ablation_scatter.pdf')

for name, sc in data.items():
    print(f"{name}: {[f'{v:.1f}' for v in sc]}, mean={np.mean(sc):.1f}")
plt.close()
