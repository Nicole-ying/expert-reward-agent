"""Fig. Budget: Best-so-Far Score vs Budget — adapted from plot_fig3_budget_matched.py"""
import json, os, numpy as np, matplotlib.pyplot as plt, matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 8,
    'axes.labelsize': 9, 'axes.titlesize': 10,
    'legend.fontsize': 7.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

BASE = Path('/home/utseus22/expert-reward-agent/runs/env_001')
THRESHOLD = 200
MAX_BUDGET = 10

# ── 1. CREATE iteration data ───────────────────────────────────
create_raw = {}
for seed_dir in sorted((BASE / 'paper_v4').glob('seed_*')):
    seed = int(seed_dir.name.split('_')[1])
    scores = []
    for it in range(1, MAX_BUDGET + 1):
        ef = seed_dir / f'iter_{it:02d}' / 'training' / 'eval_result.json'
        if ef.exists():
            scores.append(json.loads(ef.read_text(encoding='utf-8'))['mean_eval_reward'])
        elif scores:
            scores.append(scores[-1])
    if scores:
        create_raw[seed] = scores

create_bsf = {s: list(np.maximum.accumulate(sc)) for s, sc in create_raw.items()}
create_seeds = np.array([create_bsf[s] for s in sorted(create_bsf.keys())])
create_mean = create_seeds.mean(axis=0)
create_std = create_seeds.std(axis=0)
create_lower = create_mean - 1.96 * create_std / np.sqrt(len(create_seeds))
create_upper = create_mean + 1.96 * create_std / np.sqrt(len(create_seeds))

# ── 1b. Stateless Revision (Baseline) ─────────────────────────
baseline_raw = {}
for seed_dir in sorted((BASE / 'rpv4_old_baseline').glob('seed_*')):
    seed = int(seed_dir.name.split('_')[1])
    scores = []
    for it in range(1, MAX_BUDGET + 1):
        ef = seed_dir / f'iter_{it:02d}' / 'training' / 'eval_result.json'
        if ef.exists():
            scores.append(json.loads(ef.read_text(encoding='utf-8'))['mean_eval_reward'])
        elif scores:
            scores.append(scores[-1])
    if scores:
        baseline_raw[seed] = scores

baseline_bsf = {s: list(np.maximum.accumulate(sc)) for s, sc in baseline_raw.items()}
baseline_seeds = np.array([baseline_bsf[s] for s in sorted(baseline_bsf.keys())])
baseline_mean = baseline_seeds.mean(axis=0)
baseline_std = baseline_seeds.std(axis=0)
baseline_lower = baseline_mean - 1.96 * baseline_std / np.sqrt(len(baseline_seeds))
baseline_upper = baseline_mean + 1.96 * baseline_std / np.sqrt(len(baseline_seeds))

# ── 2. Independent Gen data ────────────────────────────────────
indep_dir = BASE / 'budget_matched_independent_v2'
indep_seed_scores = {}
for d in sorted(indep_dir.glob('s*_c*')):
    seed = int(d.name[1:3])
    ef = d / 'training' / 'eval_result.json'
    if ef.exists():
        score = json.loads(ef.read_text(encoding='utf-8'))['mean_eval_reward']
        indep_seed_scores.setdefault(seed, []).append(score)

N_BOOT = 100
rng = np.random.default_rng(42)
indep_bsf_all = []
for _ in range(N_BOOT):
    seed_bsf = []
    for seed in sorted(indep_seed_scores.keys()):
        scores = np.array(indep_seed_scores[seed])
        order = rng.permutation(len(scores))
        bsf = list(np.maximum.accumulate(scores[order]))
        while len(bsf) < MAX_BUDGET:
            bsf.append(bsf[-1])
        seed_bsf.append(bsf[:MAX_BUDGET])
    indep_bsf_all.append(np.mean(seed_bsf, axis=0))

indep_bsf_all = np.array(indep_bsf_all)
indep_mean = indep_bsf_all.mean(axis=0)
indep_lower = np.percentile(indep_bsf_all, 2.5, axis=0)
indep_upper = np.percentile(indep_bsf_all, 97.5, axis=0)

# ── 3. Held-out data ───────────────────────────────────────────
held_out_dir = Path(__file__).resolve().parent.parent.parent / 'analysis' / 'held_out_eval'
held_out = {
    'CREATE': json.loads(
        (held_out_dir / 'held_out_CREATE.json').read_text(encoding='utf-8')),
    'IndependentGen': json.loads(
        (held_out_dir / 'held_out_IndependentGen.json').read_text(encoding='utf-8')),
}
create_held = [v['held_out_mean'] for v in held_out['CREATE'].values()]
indep_held = [v['held_out_mean'] for v in held_out['IndependentGen'].values()]

budget_x = np.arange(1, MAX_BUDGET + 1)

out_dir = Path(__file__).resolve().parent / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────
# FIGURE (a): Best-so-Far Score vs Budget
# ────────────────────────────────────────────────────────────────
fig_a, ax_a = plt.subplots(figsize=(3.45, 2.15))

C_CREATE = '#2196F3'; C_BASELINE = '#FF9800'; C_INDEP = '#9C27B0'

# Independent Gen: mean + 95% CI
ax_a.fill_between(budget_x, indep_lower, indep_upper, alpha=0.10, color=C_INDEP)
ax_a.plot(budget_x, indep_mean, '-.', color=C_INDEP, linewidth=2.0, marker='s',
          markersize=4.0, markeredgewidth=0.6, markeredgecolor='white',
          markerfacecolor=C_INDEP, label='Independent-10', zorder=3)

# Stateless Revision: mean + 95% CI
ax_a.fill_between(budget_x, baseline_lower, baseline_upper, alpha=0.08, color=C_BASELINE)
ax_a.plot(budget_x, baseline_mean, '--', color=C_BASELINE, linewidth=2.0, marker='^',
          markersize=4.5, markeredgewidth=0.6, markeredgecolor='white',
          markerfacecolor=C_BASELINE, label='Stateless revision', zorder=4)

# CREATE: mean + 95% CI
ax_a.fill_between(budget_x, create_lower, create_upper, alpha=0.10, color=C_CREATE)
ax_a.plot(budget_x, create_mean, '-', color=C_CREATE, linewidth=2.2, marker='o',
          markersize=5.0, markeredgewidth=0.6, markeredgecolor='white',
          markerfacecolor=C_CREATE, label='CREATE', zorder=5)

# Individual seed lines (light)
for s in sorted(create_raw.keys()):
    ax_a.plot(budget_x, create_bsf[s], '-', color=C_CREATE, alpha=0.18, linewidth=0.6)
for s in sorted(baseline_raw.keys()):
    ax_a.plot(budget_x, baseline_bsf[s], '-', color=C_BASELINE, alpha=0.12, linewidth=0.5)

ax_a.axhline(y=THRESHOLD, color='#333', linestyle='--', linewidth=0.8, alpha=0.4)
ax_a.annotate(f'Threshold={THRESHOLD}', xy=(MAX_BUDGET - 0.5, THRESHOLD),
             fontsize=6.5, color='#333', ha='right', va='bottom')

ax_a.set_xlabel('Reward evaluations')
ax_a.set_ylabel('Best-so-far search fitness')
ax_a.set_xlim(0.5, MAX_BUDGET + 0.5)
ax_a.set_ylim(-135, 280)
ax_a.set_xticks(range(1, MAX_BUDGET + 1))
ax_a.legend(loc='lower right', framealpha=0.85)
ax_a.grid(axis='y', alpha=0.12)
for sp in ax_a.spines.values(): sp.set_linewidth(0.5)

fig_a.savefig(str(out_dir / 'fig2_budget_curve.pdf'))
fig_a.savefig(str(out_dir / 'fig2_budget_curve.png'), dpi=150)
print(f'Saved: {out_dir}/fig2_budget_curve.pdf')
plt.close(fig_a)

# ────────────────────────────────────────────────────────────────
# FIGURE (c): Held-out scatter
# ────────────────────────────────────────────────────────────────
fig_c, ax_c = plt.subplots(figsize=(3.45, 2.10))

rng = np.random.default_rng(123)
for i, (label, scores, c) in enumerate([
    ('CREATE', create_held, C_CREATE),
    ('Independent-10', indep_held, C_INDEP),
]):
    xs = np.full(len(scores), i) + rng.uniform(-0.14, 0.14, len(scores))
    ax_c.scatter(xs, scores, c=c, s=90, zorder=3, edgecolors='white', linewidths=0.8)
    m = np.mean(scores)
    ax_c.plot([i - 0.28, i + 0.28], [m, m], '-', color=c, linewidth=4, zorder=5,
              solid_capstyle='round')
    ax_c.annotate(f'{m:.0f}', xy=(i, m), xytext=(i, m + 22),
                  fontsize=8, ha='center', color=c, fontweight='bold')

ax_c.axhline(y=THRESHOLD, color='#333', linestyle='--', linewidth=0.8, alpha=0.4)
ax_c.annotate(f'Threshold={THRESHOLD}', xy=(1.45, THRESHOLD),
             fontsize=6.5, color='#333', va='bottom', ha='right')

ax_c.set_xticks([0, 1])
ax_c.set_xticklabels(['CREATE', 'Independent-10'])
ax_c.set_ylabel('Held-out score (100 episodes)')
ax_c.set_ylim(-80, 310)
ax_c.grid(axis='y', alpha=0.12)
for sp in ax_c.spines.values(): sp.set_linewidth(0.5)

fig_c.savefig(str(out_dir / 'fig_heldout_scatter.pdf'))
fig_c.savefig(str(out_dir / 'fig_heldout_scatter.png'), dpi=150)
print(f'Saved: {out_dir}/fig_heldout_scatter.pdf')
plt.close(fig_c)

# Print summary
print(f"\nCREATE BSF final: {create_mean[-1]:.1f} +/- {create_std[-1]:.1f}")
print(f"Baseline BSF final: {baseline_mean[-1]:.1f} +/- {baseline_std[-1]:.1f}")
print(f"Indep BSF final: {indep_mean[-1]:.1f}")
print(f"CREATE held-out mean: {np.mean(create_held):.1f}")
print(f"Indep held-out mean: {np.mean(indep_held):.1f}")
