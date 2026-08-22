"""Fig.3: Budget-matched comparison — CREATE vs Independent Generation.
(a) Best-so-Far Score vs Budget (mean + bootstrap 95% CI + individual seeds)
(b) Success@Budget (cumulative solved count)
(c) Held-out scatter
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from statistics import mean

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

BASE = Path('c:/Users/Administrator/Downloads/expert_eureka_env001_bridge_v9_direct_generator/expert_eureka_env001_bridge_v9_direct_generator/runs/env_001')
THRESHOLD = 200
MAX_BUDGET = 10

# ── 1. CREATE iteration data ───────────────────────────────────
create_bsf = {}  # seed -> [bsf_1, bsf_2, ..., bsf_10]
for seed_dir in sorted((BASE / 'paper_v4').glob('seed_*')):
    seed = int(seed_dir.name.split('_')[1])
    scores = []
    for it in range(1, MAX_BUDGET + 1):
        ef = seed_dir / f'iter_{it:02d}' / 'training' / 'eval_result.json'
        if ef.exists():
            scores.append(json.loads(ef.read_text(encoding='utf-8'))['mean_eval_reward'])
        elif scores:
            scores.append(scores[-1])  # carry forward
    if scores:
        bsf = list(np.maximum.accumulate(scores))
        create_bsf[seed] = bsf

# ── 2. Independent Gen data ────────────────────────────────────
indep_dir = BASE / 'budget_matched_independent_v2'
indep_seed_scores = {}
for d in sorted(indep_dir.glob('s*_c*')):
    seed = int(d.name[1:3])
    ef = d / 'training' / 'eval_result.json'
    if ef.exists():
        score = json.loads(ef.read_text(encoding='utf-8'))['mean_eval_reward']
        indep_seed_scores.setdefault(seed, []).append(score)

# Build best-so-far for Independent Gen (random order of candidates)
# For fair comparison: compute BSF by randomly ordering candidates, repeat N times
N_BOOT = 100
rng = np.random.default_rng(42)
indep_bsf_all = []
for _ in range(N_BOOT):
    seed_bsf = []
    for seed in sorted(indep_seed_scores.keys()):
        scores = np.array(indep_seed_scores[seed])
        order = rng.permutation(len(scores))
        bsf = list(np.maximum.accumulate(scores[order]))
        # Pad to MAX_BUDGET
        while len(bsf) < MAX_BUDGET:
            bsf.append(bsf[-1])
        seed_bsf.append(bsf[:MAX_BUDGET])
    indep_bsf_all.append(np.mean(seed_bsf, axis=0))

indep_bsf_all = np.array(indep_bsf_all)
indep_mean = indep_bsf_all.mean(axis=0)
indep_lower = np.percentile(indep_bsf_all, 2.5, axis=0)
indep_upper = np.percentile(indep_bsf_all, 97.5, axis=0)

# CREATE mean + CI
create_seeds = np.array([create_bsf[s] for s in sorted(create_bsf.keys())])
create_mean = create_seeds.mean(axis=0)
create_std = create_seeds.std(axis=0)
create_lower = create_mean - 1.96 * create_std / np.sqrt(len(create_seeds))
create_upper = create_mean + 1.96 * create_std / np.sqrt(len(create_seeds))

# ── 3. Held-out data ───────────────────────────────────────────
held_out = {
    'CREATE': json.loads(
        (Path(__file__).resolve().parent / 'held_out_eval' / 'held_out_CREATE.json')
        .read_text(encoding='utf-8')),
    'IndependentGen': json.loads(
        (Path(__file__).resolve().parent / 'held_out_eval' / 'held_out_IndependentGen.json')
        .read_text(encoding='utf-8')),
}
create_held = [v['held_out_mean'] for v in held_out['CREATE'].values()]
indep_held = [v['held_out_mean'] for v in held_out['IndependentGen'].values()]

# ── 4. Plot ────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 5))
gs = fig.add_gridspec(1, 3, wspace=0.32)

colors = {'CREATE': '#2196F3', 'Independent': '#FF5722'}
budget_x = np.arange(1, MAX_BUDGET + 1)

# --- (a) Best-so-Far Score vs Budget ---
ax_a = fig.add_subplot(gs[0, 0])

# Independent Gen: mean + 95% CI
ax_a.fill_between(budget_x, indep_lower, indep_upper, alpha=0.12, color=colors['Independent'])
ax_a.plot(budget_x, indep_mean, '-', color=colors['Independent'], linewidth=2.5, label='Independent Gen')

# CREATE: mean + 95% CI
ax_a.fill_between(budget_x, create_lower, create_upper, alpha=0.12, color=colors['CREATE'])
ax_a.plot(budget_x, create_mean, '-', color=colors['CREATE'], linewidth=2.5, label='CREATE')

# Individual CREATE seeds (light)
for s in sorted(create_bsf.keys()):
    ax_a.plot(budget_x, create_bsf[s], '-', color=colors['CREATE'], alpha=0.2, linewidth=0.8)

# Individual Indep seeds (bootstrap median seed)
# Show one representative random order per seed
for seed in sorted(indep_seed_scores.keys()):
    scores = np.array(indep_seed_scores[seed])
    order = rng.permutation(len(scores))
    bsf = list(np.maximum.accumulate(scores[order]))
    while len(bsf) < MAX_BUDGET:
        bsf.append(bsf[-1])
    ax_a.plot(budget_x, bsf[:MAX_BUDGET], '-', color=colors['Independent'], alpha=0.15, linewidth=0.8)

ax_a.axhline(y=THRESHOLD, color='#333', linestyle='--', linewidth=1, alpha=0.5)
ax_a.annotate(f'Threshold={THRESHOLD}', xy=(MAX_BUDGET-0.5, THRESHOLD),
             fontsize=7, color='#333', ha='right', va='bottom')

ax_a.set_xlabel('Budget (reward evaluations)')
ax_a.set_ylabel('Best-so-Far Score')
ax_a.set_title('(a) Best-so-Far Score vs Budget', fontweight='bold')
ax_a.legend(loc='lower right', framealpha=0.9)
ax_a.set_xlim(0.5, MAX_BUDGET + 0.5)
ax_a.grid(axis='y', alpha=0.15)

# --- (b) Success@Budget ---
ax_b = fig.add_subplot(gs[0, 1])

create_success = [sum(1 for s in sorted(create_bsf.keys())
                      if create_bsf[s][b-1] >= THRESHOLD) / len(create_bsf)
                  for b in range(1, MAX_BUDGET + 1)]
# Indep: fraction of bootstrap samples with >= 1 seed solved
indep_success = []
for b in range(1, MAX_BUDGET + 1):
    # For each bootstrap sample, check if mean bsf >= threshold at budget b
    frac = np.mean(indep_bsf_all[:, b-1] >= THRESHOLD)
    indep_success.append(frac)

ax_b.step(budget_x, create_success, '-o', where='post', color=colors['CREATE'],
          linewidth=2.5, markersize=6, label='CREATE')
ax_b.step(budget_x, indep_success, '-s', where='post', color=colors['Independent'],
          linewidth=2.5, markersize=6, label='Independent Gen')

# Annotate final values
ax_b.annotate(f'{create_success[-1]:.0%}', xy=(MAX_BUDGET, create_success[-1]),
             xytext=(MAX_BUDGET-1.5, create_success[-1]+0.08), fontsize=9,
             color=colors['CREATE'], fontweight='bold')
ax_b.annotate(f'{indep_success[-1]:.0%}', xy=(MAX_BUDGET, indep_success[-1]),
             xytext=(MAX_BUDGET-1.5, indep_success[-1]+0.08), fontsize=9,
             color=colors['Independent'], fontweight='bold')

ax_b.set_xlabel('Budget (reward evaluations)')
ax_b.set_ylabel('Success@Budget (fraction of seeds)')
ax_b.set_title('(b) Success@Budget', fontweight='bold')
ax_b.legend(loc='lower right', framealpha=0.9)
ax_b.set_xlim(0.5, MAX_BUDGET + 0.5)
ax_b.set_ylim(-0.05, 1.15)
ax_b.grid(axis='y', alpha=0.15)

# --- (c) Held-out scatter ---
ax_c = fig.add_subplot(gs[0, 2])

# Jittered scatter
rng = np.random.default_rng(123)
for i, (label, scores, c) in enumerate([
    ('CREATE', create_held, colors['CREATE']),
    ('Independent\nGen', indep_held, colors['Independent']),
]):
    xs = np.full(len(scores), i) + rng.uniform(-0.12, 0.12, len(scores))
    ax_c.scatter(xs, scores, c=c, s=80, zorder=3, edgecolors='white', linewidths=0.8,
                 label=f'{label}\n({np.mean(scores):.0f}+-{np.std(scores):.0f})')

# Mean markers
for i, (label, scores, c) in enumerate([
    ('CREATE', create_held, colors['CREATE']),
    ('IndependentGen', indep_held, colors['Independent']),
]):
    m = np.mean(scores)
    ax_c.plot([i-0.28, i+0.28], [m, m], '-', color=c, linewidth=4, zorder=5, solid_capstyle='round')
    ax_c.annotate(f'{m:.1f}', xy=(i, m), xytext=(i, m + 18),
                  fontsize=8, ha='center', color=c, fontweight='bold')

ax_c.axhline(y=THRESHOLD, color='#333', linestyle='--', linewidth=1, alpha=0.5)
ax_c.annotate(f'Threshold={THRESHOLD}', xy=(1.4, THRESHOLD),
             fontsize=7, color='#333', va='bottom', ha='right')

ax_c.set_xticks([0, 1])
ax_c.set_xticklabels(['CREATE', 'Independent\nGeneration'])
ax_c.set_ylabel('Held-out Score (100 episodes)')
ax_c.set_title('(c) Held-out Generalization', fontweight='bold')
ax_c.legend(loc='lower right', fontsize=7, framealpha=0.8)
ax_c.grid(axis='y', alpha=0.15)

fig.suptitle('Fig.3: Budget-Matched Comparison — CREATE vs Independent Generation (LunarLander-v3)',
             fontweight='bold', fontsize=12, y=1.01)

out_dir = Path(__file__).resolve().parent
fig.savefig(str(out_dir / 'fig3_budget_matched.pdf'))
fig.savefig(str(out_dir / 'fig3_budget_matched.png'), dpi=150)
print(f'Saved: fig3_budget_matched.pdf and .png')
plt.close()
