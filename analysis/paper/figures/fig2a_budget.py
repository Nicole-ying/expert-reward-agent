"""Fig 2: Best-so-Far Score vs Budget — standalone, based on original plot_fig3"""
import json, os, numpy as np, matplotlib.pyplot as plt, matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

BASE = Path('/home/utseus22/expert-reward-agent/runs/env_001')
THRESHOLD = 200; MAX_BUDGET = 10

# ── CREATE ──
create_bsf = {}
for seed_dir in sorted((BASE / 'paper_v4').glob('seed_*')):
    seed = int(seed_dir.name.split('_')[1])
    scores = []
    for it in range(1, MAX_BUDGET + 1):
        ef = seed_dir / f'iter_{it:02d}' / 'training' / 'eval_result.json'
        if ef.exists(): scores.append(json.loads(ef.read_text(encoding='utf-8'))['mean_eval_reward'])
        elif scores: scores.append(scores[-1])
    if scores: create_bsf[seed] = list(np.maximum.accumulate(scores))

# ── Stateless Revision ──
baseline_bsf = {}
for seed_dir in sorted((BASE / 'rpv4_old_baseline').glob('seed_*')):
    seed = int(seed_dir.name.split('_')[1])
    scores = []
    for it in range(1, MAX_BUDGET + 1):
        ef = seed_dir / f'iter_{it:02d}' / 'training' / 'eval_result.json'
        if ef.exists(): scores.append(json.loads(ef.read_text(encoding='utf-8'))['mean_eval_reward'])
        elif scores: scores.append(scores[-1])
    if scores: baseline_bsf[seed] = list(np.maximum.accumulate(scores))

# ── Independent Gen ──
indep_dir = BASE / 'budget_matched_independent_v2'
indep_seed_scores = {}
for d in sorted(indep_dir.glob('s*_c*')):
    seed = int(d.name[1:3])
    ef = d / 'training' / 'eval_result.json'
    if ef.exists():
        indep_seed_scores.setdefault(seed, []).append(json.loads(ef.read_text(encoding='utf-8'))['mean_eval_reward'])

N_BOOT = 100; rng = np.random.default_rng(42)
indep_bsf_all = []
for _ in range(N_BOOT):
    seed_bsf = []
    for seed in sorted(indep_seed_scores.keys()):
        scores = np.array(indep_seed_scores[seed])
        order = rng.permutation(len(scores))
        bsf = list(np.maximum.accumulate(scores[order]))
        while len(bsf) < MAX_BUDGET: bsf.append(bsf[-1])
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

# Baseline mean + CI
baseline_seeds = np.array([baseline_bsf[s] for s in sorted(baseline_bsf.keys())])
baseline_mean = baseline_seeds.mean(axis=0)
baseline_std = baseline_seeds.std(axis=0)
baseline_lower = baseline_mean - 1.96 * baseline_std / np.sqrt(len(baseline_seeds))
baseline_upper = baseline_mean + 1.96 * baseline_std / np.sqrt(len(baseline_seeds))

budget_x = np.arange(1, MAX_BUDGET + 1)

# ── Plot: EXACT same calls as original subplot (a), just standalone ──
fig, ax = plt.subplots(figsize=(4.5, 3.5))

colors = {'CREATE': '#2196F3', 'Independent': '#FF5722'}

# Independent Gen
ax.fill_between(budget_x, indep_lower, indep_upper, alpha=0.12, color=colors['Independent'])
ax.plot(budget_x, indep_mean, '-', color=colors['Independent'], linewidth=2.5, label='Independent Gen')
for seed in sorted(indep_seed_scores.keys()):
    scores = np.array(indep_seed_scores[seed])
    order = rng.permutation(len(scores))
    bsf = list(np.maximum.accumulate(scores[order]))
    while len(bsf) < MAX_BUDGET: bsf.append(bsf[-1])
    ax.plot(budget_x, bsf[:MAX_BUDGET], '-', color=colors['Independent'], alpha=0.15, linewidth=0.8)

# Stateless Revision
ax.fill_between(budget_x, baseline_lower, baseline_upper, alpha=0.10, color='#FF9800')
ax.plot(budget_x, baseline_mean, '-', color='#FF9800', linewidth=2.2, label='Stateless revision')
for s in sorted(baseline_bsf.keys()):
    ax.plot(budget_x, baseline_bsf[s], '-', color='#FF9800', alpha=0.15, linewidth=0.7)

# CREATE
ax.fill_between(budget_x, create_lower, create_upper, alpha=0.12, color=colors['CREATE'])
ax.plot(budget_x, create_mean, '-', color=colors['CREATE'], linewidth=2.5, label='CREATE')
for s in sorted(create_bsf.keys()):
    ax.plot(budget_x, create_bsf[s], '-', color=colors['CREATE'], alpha=0.2, linewidth=0.8)

ax.axhline(y=THRESHOLD, color='#333', linestyle='--', linewidth=1, alpha=0.5)
ax.annotate(f'Threshold={THRESHOLD}', xy=(MAX_BUDGET - 0.5, THRESHOLD),
             fontsize=7, color='#333', ha='right', va='bottom')

ax.set_xlabel('Budget (reward evaluations)')
ax.set_ylabel('Best-so-far search fitness')
ax.legend(loc='lower right', framealpha=0.9)
ax.set_xlim(0.5, MAX_BUDGET + 0.5)
ax.grid(axis='y', alpha=0.15)

out = Path(__file__).resolve().parent / 'figures' / 'fig2_budget_curve.pdf'
fig.savefig(str(out)); fig.savefig(str(out.with_suffix('.png')), dpi=150)
print(f'Saved: {out}'); plt.close()
