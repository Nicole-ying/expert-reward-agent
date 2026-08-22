"""Held-out scatter — original plot_fig3 subplot(c) style + black mean bars + small dots."""
import json, numpy as np, matplotlib.pyplot as plt, matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

THRESHOLD = 200
held_out_dir = Path(__file__).resolve().parent.parent.parent / 'analysis' / 'held_out_eval'
held_out = {
    'CREATE': json.loads((held_out_dir / 'held_out_CREATE.json').read_text(encoding='utf-8')),
    'Stateless': json.loads((held_out_dir / 'held_out_StatelessRevision.json').read_text(encoding='utf-8')),
    'IndependentGen': json.loads((held_out_dir / 'held_out_IndependentGen.json').read_text(encoding='utf-8')),
}
create_held = [v['held_out_mean'] for v in held_out['CREATE'].values()]
stateless_held = [v['held_out_mean'] for v in held_out['Stateless'].values()]
indep_held = [v['held_out_mean'] for v in held_out['IndependentGen'].values()]

fig, ax = plt.subplots(figsize=(5.5, 3.5))
colors = {'CREATE': '#2196F3', 'Stateless': '#FF9800', 'Independent': '#FF5722'}
rng = np.random.default_rng(123)

for i, (label, scores, c, jw) in enumerate([
    ('Independent\nGen', indep_held, colors['Independent'], 0.12),
    ('Stateless\nrevision', stateless_held, colors['Stateless'], 0.12),
    ('CREATE', create_held, colors['CREATE'], 0.12),
]):
    xs = np.full(len(scores), i) + rng.uniform(-jw, jw, len(scores))
    # Manually separate the two nearly-identical Independent seeds (-109.4 and -109.2)
    if label == 'Independent\nGen':
        xs[3] = i + 0.10   # seed_3 right
        xs[4] = i - 0.10   # seed_4 left
    ax.scatter(xs, scores, c=c, s=55, zorder=3, edgecolors='white', linewidths=0.6,
               label=f'{label}\n({np.mean(scores):.0f}±{np.std(scores):.0f})')

for i, (label, scores, c, _) in enumerate([
    ('IndependentGen', indep_held, colors['Independent'], 0),
    ('Stateless', stateless_held, colors['Stateless'], 0),
    ('CREATE', create_held, colors['CREATE'], 0),
]):
    m = np.mean(scores)
    ax.plot([i-0.15, i+0.15], [m, m], '-', color='black', linewidth=1.2,
            zorder=5, solid_capstyle='round')

ax.axhline(y=THRESHOLD, color='#333', linestyle='--', linewidth=0.8, alpha=0.4)
ax.annotate(f'Threshold={THRESHOLD}', xy=(2.35, THRESHOLD),
             fontsize=7, color='#333', va='bottom', ha='right')

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Independent\nGeneration', 'Stateless\nrevision', 'CREATE'])
ax.set_ylabel('Test fitness')
ax.legend(loc='lower right', fontsize=7, framealpha=0.8)
ax.grid(axis='y', alpha=0.15)

out = Path(__file__).resolve().parent / 'figures' / 'fig_heldout_scatter.pdf'
fig.savefig(str(out)); fig.savefig(str(out.with_suffix('.png')), dpi=150)
print(f'Saved: {out}'); plt.close()
