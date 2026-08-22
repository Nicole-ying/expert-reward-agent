"""Fig.4: Ablation paired scatter plot — 4 methods × 5 seeds on LunarLander-v3."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── Data ────────────────────────────────────────────────────────
# (LLM-once, Coarse, Unconstrained, CREATE) per seed
methods = ['LLM-once\n(iter_01)', 'Coarse\nFeedback', 'Unconstrained\nRefinement', 'CREATE\n(full)']
data = {
    'LLM-once':        [-70.35, -42.74, -17.90, -19.59, 139.53],
    'CoarseFeedback':  [239.52, 170.40, -110.09, 115.51, 259.50],
    'Unconstrained':   [169.90, 130.64,  71.06,  59.18, 140.27],
    'CREATE':          [224.21, 240.60, 220.24, 253.71, 206.14],
}
n_seeds = 5
n_methods = 4
THRESHOLD = 200

seed_colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd']
method_colors = ['#bdbdbd', '#ff9800', '#f44336', '#4caf50']

# ── Plot ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.5))

# Paired lines
for s in range(n_seeds):
    scores = [data['LLM-once'][s], data['CoarseFeedback'][s],
              data['Unconstrained'][s], data['CREATE'][s]]
    ax.plot(range(n_methods), scores, '-', color=seed_colors[s],
            alpha=0.4, linewidth=1.2, zorder=1)

# Per-method scatter (with jitter)
for m, (method_name, key) in enumerate(zip(methods, data.keys())):
    xs = np.full(n_seeds, m) + np.random.default_rng(42).uniform(-0.08, 0.08, n_seeds)
    ax.scatter(xs, data[key], c=seed_colors, s=60, zorder=3, edgecolors='white', linewidths=0.5)

# Method mean markers
for m, key in enumerate(data.keys()):
    mean_val = np.mean(data[key])
    ax.plot([m-0.25, m+0.25], [mean_val, mean_val], '-',
            color=method_colors[m], linewidth=3, zorder=4, solid_capstyle='round')
    # Mean label
    ax.annotate(f'{mean_val:.0f}', xy=(m, mean_val),
                xytext=(m, mean_val + 22 if mean_val > 0 else mean_val - 30),
                fontsize=7.5, ha='center', color=method_colors[m], fontweight='bold')

# Threshold line
ax.axhline(y=THRESHOLD, color='#333', linestyle='--', linewidth=1, alpha=0.6, zorder=0)
ax.annotate(f'Solved threshold ({THRESHOLD})', xy=(3.4, THRESHOLD),
            xytext=(3.4, THRESHOLD+15), fontsize=7.5, ha='right', color='#333')

# Solved count annotations
solved_counts = {
    'LLM-once': 0, 'CoarseFeedback': 2, 'Unconstrained': 0, 'CREATE': 5
}
for m, key in enumerate(data.keys()):
    ax.annotate(f'{solved_counts[key]}/5 solved',
                xy=(m, ax.get_ylim()[0] if ax.get_ylim()[0] > -200 else -170),
                fontsize=7.5, ha='center', color=method_colors[m], fontweight='bold')

# Styling
ax.set_xticks(range(n_methods))
ax.set_xticklabels(methods)
ax.set_ylabel('Development Score (original env reward)')
ax.set_ylim(-190, 310)
ax.grid(axis='y', alpha=0.2, zorder=0)

# Legend for seed colors
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                          markersize=8, label=f'seed_{i}')
                   for i, c in enumerate(seed_colors)]
ax.legend(handles=legend_elements, loc='lower right', ncol=5, framealpha=0.8)

ax.set_title('Ablation: Development Scores by Method and Seed', fontweight='bold', pad=12)

out = Path(__file__).resolve().parent / 'fig4_ablation_paired.pdf'
fig.savefig(str(out))
png_out = Path(__file__).resolve().parent / 'fig4_ablation_paired.png'
fig.savefig(str(png_out), dpi=150)
print(f'Saved: {out}')
print(f'Saved: {png_out}')
plt.close()
