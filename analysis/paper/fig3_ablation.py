"""Fig.3: Ablation scatter — matching heldout style: legend+mean±std, black bar, small dots."""
import json, matplotlib.pyplot as plt, matplotlib, numpy as np
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

def load_best(base):
    scores = []
    for s in range(5):
        f = Path(base) / f'seed_{s}' / 'best' / 'best_training_summary.json'
        if f.exists():
            scores.append(json.loads(f.read_text(encoding='utf-8'))['external_eval']['mean_eval_reward'])
    return scores

BASE = '/home/utseus22/expert-reward-agent/runs/env_001'
data = {
    'w/o Evidence':  ('#FF9800', load_best(BASE + '/ablation_eureka_feedback_v4')),
    'w/o Hierarchy': ('#9C27B0', load_best(BASE + '/ablation_unconstrained_v4')),
    'CREATE':        ('#2196F3', load_best(BASE + '/paper_v4')),
}
groups = list(data.keys())
THRESHOLD = 200
rng = np.random.default_rng(42)

fig, ax = plt.subplots(figsize=(5.5, 3.5))

for i, (name, (color, scores)) in enumerate(data.items()):
    arr = np.array(scores)
    mu, sd = arr.mean(), arr.std()
    xs = np.full(5, i) + rng.uniform(-0.12, 0.12, 5)
    ax.scatter(xs, scores, c=color, s=55, zorder=3, edgecolors='white', linewidths=0.6,
               label=f'{name}\n({mu:.0f}±{sd:.0f})', clip_on=False)

    # Black mean bar
    ax.plot([i-0.15, i+0.15], [mu, mu], '-', color='black', linewidth=1.2,
            zorder=5, solid_capstyle='round')

ax.axhline(y=THRESHOLD, color='#333', linestyle='--', linewidth=0.8, alpha=0.4)
ax.annotate(f'Threshold={THRESHOLD}', xy=(2.35, THRESHOLD),
             fontsize=7, color='#333', va='bottom', ha='right')

ax.set_xticks(range(3))
ax.set_xticklabels(groups)
ax.set_ylabel('Best search fitness (lineage)')
ax.set_ylim(-195, 310)
ax.legend(loc='lower right', fontsize=7, framealpha=0.8)
ax.grid(axis='y', alpha=0.15)
for sp in ax.spines.values(): sp.set_linewidth(0.5)

out = Path(__file__).resolve().parent / 'figures' / 'fig3_ablation_scatter.pdf'
fig.savefig(str(out)); fig.savefig(str(out.with_suffix('.png')), dpi=150)
print(f'Saved: {out}'); plt.close()
