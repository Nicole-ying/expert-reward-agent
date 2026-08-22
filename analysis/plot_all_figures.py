"""Generate all paper figures as individual, standalone files.

Outputs (all PDF + PNG):
  fig4_ablation.pdf          — 4 methods scatter + mean bar + threshold
  fig3a_bsf_vs_budget.pdf    — Best-so-Far Score vs Budget
  fig3b_success_at_budget.pdf— Success@Budget
  fig3c_held_out_scatter.pdf — Held-out generalization scatter
  fig5a_score_trajectory.pdf — Seed_0 score over iterations
  fig5b_activation_heatmap.pdf — Component activation rate
  fig5c_magnitude_heatmap.pdf  — Component magnitude share
"""
import json, re, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from statistics import mean

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

OUT = Path(__file__).resolve().parent
BASE = Path('c:/Users/Administrator/Downloads/expert_eureka_env001_bridge_v9_direct_generator/expert_eureka_env001_bridge_v9_direct_generator/runs/env_001')
THRESHOLD = 200
MAX_BUDGET = 10
SEED_COLORS = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd']
CREATE_C = '#2196F3'
INDEP_C = '#FF5722'

def save(fig, name):
    fig.savefig(str(OUT / f'{name}.pdf'))
    fig.savefig(str(OUT / f'{name}.png'), dpi=150)
    print(f'  {name}.pdf + .png')

# ====================================================================
# Fig.4: Ablation scatter (NO connecting lines — independent methods)
# ====================================================================
def make_fig4():
    methods = ['LLM-once', 'Coarse\nFeedback', 'Unconstrained\nRefinement', 'CREATE']
    data = {
        'LLM-once':        [-70.35, -42.74, -17.90, -19.59, 139.53],
        'CoarseFeedback':  [239.52, 170.40, -110.09, 115.51, 259.50],
        'Unconstrained':   [169.90, 130.64,  71.06,  59.18, 140.27],
        'CREATE':          [224.21, 240.60, 220.24, 253.71, 206.14],
    }
    solved = {'LLM-once': 0, 'CoarseFeedback': 2, 'Unconstrained': 0, 'CREATE': 5}
    method_colors = ['#9e9e9e', '#FF9800', '#f44336', '#4CAF50']
    n_methods = len(methods)
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for m, (method_name, key) in enumerate(zip(methods, data.keys())):
        xs = np.full(5, m) + rng.uniform(-0.1, 0.1, 5)
        ax.scatter(xs, data[key], c=SEED_COLORS, s=70, zorder=3,
                   edgecolors='white', linewidths=0.5)

        # Mean bar
        mval = np.mean(data[key])
        ax.plot([m-0.3, m+0.3], [mval, mval], '-', color=method_colors[m],
                linewidth=3.5, zorder=4, solid_capstyle='round')
        # Mean label
        offset = 25 if mval > 0 else -30
        ax.annotate(f'{mval:.1f}', xy=(m, mval), xytext=(m, mval + offset),
                    fontsize=8.5, ha='center', color=method_colors[m], fontweight='bold')

        # Solved count below
        ax.annotate(f'{solved[key]}/5', xy=(m, -185), fontsize=8, ha='center',
                    color=method_colors[m], fontweight='bold')

    # Threshold
    ax.axhline(y=THRESHOLD, color='#333', linestyle='--', linewidth=1, alpha=0.5)
    ax.annotate(f'Threshold = {THRESHOLD}', xy=(3.4, THRESHOLD),
                xytext=(3.4, THRESHOLD + 18), fontsize=8, ha='right', color='#333')

    # Seed legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8,
                      label=f'seed {i}') for i, c in enumerate(SEED_COLORS)]
    ax.legend(handles=handles, loc='lower left', ncol=5, framealpha=0.8, fontsize=8)

    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(methods)
    ax.set_ylabel('Development Score')
    ax.set_ylim(-195, 310)
    ax.grid(axis='y', alpha=0.15)
    ax.set_title('Ablation: Development Scores by Method (LunarLander-v3)', fontweight='bold')

    save(fig, 'fig4_ablation')

# ====================================================================
# Fig.3a: Best-so-Far Score vs Budget
# ====================================================================
def make_fig3a():
    # Load CREATE data
    create_bsf = {}
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
            create_bsf[seed] = list(np.maximum.accumulate(scores))

    # Load Independent Gen data
    indep_dir = BASE / 'budget_matched_independent_v2'
    indep_seed_scores = {}
    for d in sorted(indep_dir.glob('s*_c*')):
        seed = int(d.name[1:3])
        ef = d / 'training' / 'eval_result.json'
        if ef.exists():
            indep_seed_scores.setdefault(seed, []).append(
                json.loads(ef.read_text(encoding='utf-8'))['mean_eval_reward'])

    rng = np.random.default_rng(42)
    indep_bsf_all = []
    for _ in range(200):
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

    create_seeds = np.array([create_bsf[s] for s in sorted(create_bsf.keys())])
    create_mean = create_seeds.mean(axis=0)

    budget_x = np.arange(1, MAX_BUDGET + 1)

    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Independent Gen CI
    ax.fill_between(budget_x,
                    np.percentile(indep_bsf_all, 2.5, axis=0),
                    np.percentile(indep_bsf_all, 97.5, axis=0),
                    alpha=0.12, color=INDEP_C)
    ax.plot(budget_x, indep_bsf_all.mean(axis=0), '-', color=INDEP_C, linewidth=2.5,
            label='Independent Generation')

    # CREATE CI
    create_std = create_seeds.std(axis=0)
    ax.fill_between(budget_x,
                    create_mean - 1.96*create_std/np.sqrt(len(create_seeds)),
                    create_mean + 1.96*create_std/np.sqrt(len(create_seeds)),
                    alpha=0.12, color=CREATE_C)
    ax.plot(budget_x, create_mean, '-', color=CREATE_C, linewidth=2.5, label='CREATE')

    # Individual seeds (light)
    for s in sorted(create_bsf.keys()):
        ax.plot(budget_x, create_bsf[s], '-', color=CREATE_C, alpha=0.18, linewidth=0.7)

    ax.axhline(y=THRESHOLD, color='#333', linestyle='--', linewidth=1, alpha=0.5)
    ax.annotate(f'Threshold = {THRESHOLD}', xy=(9, THRESHOLD),
                fontsize=8, color='#333', ha='right', va='bottom')

    ax.set_xlabel('Budget (reward evaluations)')
    ax.set_ylabel('Best-so-Far Score')
    ax.set_title('Best-so-Far Score vs Budget', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(0.5, MAX_BUDGET + 0.5)
    ax.grid(axis='y', alpha=0.15)

    save(fig, 'fig3a_bsf_vs_budget')

# ====================================================================
# Fig.3b: Success@Budget
# ====================================================================
def make_fig3b():
    create_bsf = {}
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
            create_bsf[seed] = list(np.maximum.accumulate(scores))

    indep_dir = BASE / 'budget_matched_independent_v2'
    indep_seed_scores = {}
    for d in sorted(indep_dir.glob('s*_c*')):
        seed = int(d.name[1:3])
        ef = d / 'training' / 'eval_result.json'
        if ef.exists():
            indep_seed_scores.setdefault(seed, []).append(
                json.loads(ef.read_text(encoding='utf-8'))['mean_eval_reward'])

    rng = np.random.default_rng(42)
    indep_bsf_all = []
    for _ in range(200):
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

    budget_x = np.arange(1, MAX_BUDGET + 1)

    create_succ = [sum(1 for s in sorted(create_bsf.keys())
                       if create_bsf[s][b-1] >= THRESHOLD) / len(create_bsf)
                   for b in range(1, MAX_BUDGET + 1)]
    indep_succ = [np.mean(indep_bsf_all[:, b-1] >= THRESHOLD) for b in range(1, MAX_BUDGET + 1)]

    fig, ax = plt.subplots(figsize=(5.5, 4))

    ax.step(budget_x, create_succ, '-o', where='post', color=CREATE_C,
            linewidth=2.5, markersize=7, label='CREATE')
    ax.step(budget_x, indep_succ, '-s', where='post', color=INDEP_C,
            linewidth=2.5, markersize=7, label='Independent Generation')

    ax.annotate(f'{create_succ[-1]:.0%}', xy=(MAX_BUDGET, create_succ[-1]),
                xytext=(MAX_BUDGET-2, create_succ[-1]+0.08), fontsize=11,
                color=CREATE_C, fontweight='bold')
    ax.annotate(f'{indep_succ[-1]:.0%}', xy=(MAX_BUDGET, indep_succ[-1]),
                xytext=(MAX_BUDGET-2, indep_succ[-1]+0.08), fontsize=11,
                color=INDEP_C, fontweight='bold')

    ax.set_xlabel('Budget (reward evaluations)')
    ax.set_ylabel('Success@Budget')
    ax.set_title('Success@Budget', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(0.5, MAX_BUDGET + 0.5)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(axis='y', alpha=0.15)

    save(fig, 'fig3b_success_at_budget')

# ====================================================================
# Fig.3c: Held-out scatter
# ====================================================================
def make_fig3c():
    held_out = {
        'CREATE': json.loads((OUT / 'held_out_eval' / 'held_out_CREATE.json').read_text('utf-8')),
        'Independent': json.loads((OUT / 'held_out_eval' / 'held_out_IndependentGen.json').read_text('utf-8')),
    }
    create_h = [v['held_out_mean'] for v in held_out['CREATE'].values()]
    indep_h = [v['held_out_mean'] for v in held_out['Independent'].values()]
    rng = np.random.default_rng(123)

    fig, ax = plt.subplots(figsize=(5, 4))

    for i, (label, scores, c) in enumerate([
        ('CREATE', create_h, CREATE_C),
        ('Independent Gen', indep_h, INDEP_C),
    ]):
        xs = np.full(len(scores), i) + rng.uniform(-0.12, 0.12, len(scores))
        ax.scatter(xs, scores, c=c, s=100, zorder=3, edgecolors='white', linewidths=0.8)

        m = np.mean(scores)
        ax.plot([i-0.3, i+0.3], [m, m], '-', color=c, linewidth=4, zorder=5, solid_capstyle='round')
        ax.annotate(f'{m:.1f}', xy=(i, m), xytext=(i, m + 25),
                    fontsize=10, ha='center', color=c, fontweight='bold')
        ax.annotate(f'{sum(1 for s in scores if s >= THRESHOLD)}/5 solved',
                    xy=(i, -155), fontsize=8.5, ha='center', color=c, fontweight='bold')

    ax.axhline(y=THRESHOLD, color='#333', linestyle='--', linewidth=1, alpha=0.5)
    ax.annotate(f'Threshold = {THRESHOLD}', xy=(1.4, THRESHOLD),
                fontsize=8, color='#333', va='bottom', ha='right')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['CREATE', 'Independent Generation'])
    ax.set_ylabel('Held-out Score (100 episodes)')
    ax.set_ylim(-170, 310)
    ax.set_title('Held-out Generalization', fontweight='bold')
    ax.grid(axis='y', alpha=0.15)

    save(fig, 'fig3c_held_out_scatter')

# ====================================================================
# Fig.5a: Score trajectory
# ====================================================================
def make_fig5a():
    seed_dir = BASE / 'paper_v4' / 'seed_0'
    iters, scores, edits = [], [], []
    for d in sorted(seed_dir.glob('iter_*')):
        it = int(d.name.split('_')[1])
        ef = d / 'training' / 'eval_result.json'
        rf = d / 'generation' / 'response_records' / 'agent_reflection.md'
        if not ef.exists():
            continue
        score = json.loads(ef.read_text(encoding='utf-8'))['mean_eval_reward']
        # Extract edit level
        level = '?'
        if rf.exists():
            t = rf.read_text(encoding='utf-8')
            m = re.search(r'Level\s+(\d)', t)
            if m: level = f'L{m.group(1)}'
        if it == 1 and level == '?': level = 'initial'
        iters.append(it); scores.append(score); edits.append(level)

    bsf = list(np.maximum.accumulate(scores))
    edit_colors = {'L1': '#4CAF50', 'L2': '#FF9800', 'L3': '#f44336', 'initial': '#9e9e9e', '?': '#bbb'}

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, scores, '-o', color='#1f77b4', linewidth=2, markersize=9,
            markerfacecolor='white', markeredgewidth=2, zorder=3, label='Per-iteration score')
    ax.plot(iters, bsf, '--', color='#FF9800', linewidth=2, alpha=0.85,
            label='Best-so-far (Best Archive)')

    for it, sc, ed in zip(iters, scores, edits):
        c = edit_colors.get(ed, '#999')
        offset = 35 if sc < 0 else 25
        ax.annotate(ed, xy=(it, sc), xytext=(it, sc + offset),
                    fontsize=8, ha='center', color=c, fontweight='bold')

    ax.axhline(y=THRESHOLD, color='#333', linestyle='--', linewidth=1, alpha=0.5)
    ax.annotate(f'Threshold = {THRESHOLD}', xy=(iters[-1], THRESHOLD),
                fontsize=8, color='#333', ha='right', va='bottom')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Development Score')
    ax.set_title('Score Trajectory — CREATE seed_0 (LunarLander-v3)', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(iters[0]-0.5, iters[-1]+0.5)
    ax.grid(axis='y', alpha=0.15)

    save(fig, 'fig5a_score_trajectory')

# ====================================================================
# Fig.5b+c: Component heatmaps
# ====================================================================
def make_fig5bc():
    seed_dir = BASE / 'paper_v4' / 'seed_0'
    iter_data = {}
    for d in sorted(seed_dir.glob('iter_*')):
        it = int(d.name.split('_')[1])
        ts = d / 'training' / 'training_summary.json'
        if not ts.exists(): continue
        train = json.loads(ts.read_text(encoding='utf-8'))
        comp_stats = train.get('component_summary', {}).get('component_stats', {})
        ep_comp = train.get('component_summary', {}).get('episode_component_sum_stats', {})

        comps = {}
        total_abs = sum(abs(ep_comp.get(c.replace('component.', ''), {}).get('abs_mean', 0))
                        for c in comp_stats if 'component.' in c)
        for name, item in comp_stats.items():
            short = name.replace('component.', '')
            if short in ('generated_reward', 'total_reward', 'original_env_reward'):
                continue
            ep_abs = abs(ep_comp.get(short, {}).get('abs_mean', 0))
            comps[short] = {
                'active_rate': item.get('nonzero_rate', 0),
                'mag_share': ep_abs / total_abs if total_abs > 1e-12 else 0,
            }
        iter_data[it] = comps

    iters = sorted(iter_data.keys())
    all_comps = sorted(set(c for i in iters for c in iter_data[i]))

    # (b) Activation rate
    fig_b, ax_b = plt.subplots(figsize=(7, max(3, len(all_comps)*0.35)))
    act = np.zeros((len(all_comps), len(iters)))
    for ci, comp in enumerate(all_comps):
        for ti, it in enumerate(iters):
            act[ci, ti] = iter_data[it].get(comp, {}).get('active_rate', 0)

    im = ax_b.imshow(act, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax_b.set_xticks(range(len(iters))); ax_b.set_xticklabels([str(i) for i in iters])
    ax_b.set_yticks(range(len(all_comps))); ax_b.set_yticklabels(all_comps, fontsize=7)
    ax_b.set_title('Component Activation Rate (seed_0)', fontweight='bold')
    ax_b.set_xlabel('Iteration')
    for ci in range(len(all_comps)):
        for ti in range(len(iters)):
            v = act[ci, ti]
            ax_b.text(ti, ci, f'{v:.2f}', ha='center', va='center',
                      fontsize=6.5, color='white' if v > 0.5 else '#333')
    plt.colorbar(im, ax=ax_b, shrink=0.85)
    save(fig_b, 'fig5b_activation_heatmap')

    # (c) Magnitude share
    fig_c, ax_c = plt.subplots(figsize=(7, max(3, len(all_comps)*0.35)))
    mag = np.zeros((len(all_comps), len(iters)))
    for ci, comp in enumerate(all_comps):
        for ti, it in enumerate(iters):
            mag[ci, ti] = iter_data[it].get(comp, {}).get('mag_share', 0)

    im2 = ax_c.imshow(mag, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax_c.set_xticks(range(len(iters))); ax_c.set_xticklabels([str(i) for i in iters])
    ax_c.set_yticks(range(len(all_comps))); ax_c.set_yticklabels(all_comps, fontsize=7)
    ax_c.set_title('Component Magnitude Share (seed_0)', fontweight='bold')
    ax_c.set_xlabel('Iteration')
    for ci in range(len(all_comps)):
        for ti in range(len(iters)):
            v = mag[ci, ti]
            ax_c.text(ti, ci, f'{v:.1%}', ha='center', va='center',
                      fontsize=6.5, color='white' if v > 0.5 else '#333')
    plt.colorbar(im2, ax=ax_c, shrink=0.85)
    save(fig_c, 'fig5c_magnitude_heatmap')

# ====================================================================
if __name__ == '__main__':
    print("Generating all figures...")
    make_fig4()
    make_fig3a()
    make_fig3b()
    make_fig3c()
    make_fig5a()
    make_fig5bc()
    print("\nAll figures saved to analysis/")
