"""Fig.5: Case study triple analysis — LunarLander seed_0 trajectory."""
import json, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import OrderedDict

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 8,
    'axes.labelsize': 9, 'axes.titlesize': 10,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

BASE = Path('c:/Users/Administrator/Downloads/expert_eureka_env001_bridge_v9_direct_generator/expert_eureka_env001_bridge_v9_direct_generator/runs/env_001/paper_v4/seed_0')

THRESHOLD = 200

# ── 1. Read iteration data ─────────────────────────────────────
iterations = {}
for d in sorted(BASE.glob('iter_*')):
    it = int(d.name.split('_')[1])
    ts = d / 'training' / 'training_summary.json'
    es = d / 'training' / 'eval_result.json'
    gm = d / 'generation' / 'prompt_records' / '03_reward_revision.md'
    if not ts.exists() or not es.exists():
        continue

    train = json.loads(ts.read_text(encoding='utf-8'))
    ev = json.loads(es.read_text(encoding='utf-8'))

    comp_stats = train.get('component_summary', {}).get('component_stats', {})
    ep_comp = train.get('component_summary', {}).get('episode_component_sum_stats', {})

    # Parse edit level from revision prompt
    edit_level = '?'
    if gm.exists():
        text = gm.read_text(encoding='utf-8')
        if 'L1' in text and '参数调整' in text:
            edit_level = 'L1'
        elif 'L2' in text and ('组件重构' in text or '重构' in text):
            edit_level = 'L2'
        elif 'L3' in text and ('重设计' in text or 'redesign' in text.lower()):
            edit_level = 'L3'
        elif '初始' in text or 'initial' in text.lower() or it == 1:
            edit_level = 'initial'

    # Extract component data
    comps = OrderedDict()
    for name, item in sorted(comp_stats.items()):
        short = name.replace('component.', '')
        if short in {'generated_reward', 'total_reward', 'original_env_reward'}:
            continue
        comps[short] = {
            'mean': item.get('mean', 0),
            'abs_mean': item.get('abs_mean', 0),
            'nonzero_rate': item.get('nonzero_rate', 0),
            'ep_sum_mean': ep_comp.get(short, {}).get('mean', 0) if short in ep_comp else 0,
            'ep_abs_sum_mean': ep_comp.get(short, {}).get('abs_mean', 0) if short in ep_comp else 0,
        }

    iterations[it] = {
        'score': ev['mean_eval_reward'],
        'score_min': ev['min_eval_reward'],
        'score_max': ev['max_eval_reward'],
        'edit': edit_level,
        'components': comps,
    }

if not iterations:
    print("ERROR: No iteration data found!")
    exit(1)

iters = sorted(iterations.keys())
scores = [iterations[i]['score'] for i in iters]
edits = [iterations[i]['edit'] for i in iters]

# Collect all component names across iterations
all_comps = set()
for i in iters:
    all_comps.update(iterations[i]['components'].keys())
all_comps = sorted(all_comps)

# ── 2. Plot ────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.30, wspace=0.28)

# --- (a) Score trajectory ---
ax_a = fig.add_subplot(gs[0, :])
ax_a.plot(iters, scores, '-o', color='#1f77b4', linewidth=2, markersize=8,
          markerfacecolor='white', markeredgewidth=2, zorder=3)
ax_a.fill_between(iters, [iterations[i]['score_min'] for i in iters],
                  [iterations[i]['score_max'] for i in iters],
                  alpha=0.15, color='#1f77b4')

# Best-so-far
best_sofar = np.maximum.accumulate(scores)
ax_a.plot(iters, best_sofar, '--', color='#ff7f0e', linewidth=1.5, alpha=0.8,
          label='Best-so-far (Best Archive)')

# Threshold
ax_a.axhline(y=THRESHOLD, color='#333', linestyle='--', linewidth=1, alpha=0.5)
ax_a.annotate(f'Threshold={THRESHOLD}', xy=(iters[-1]+0.2, THRESHOLD),
             fontsize=7, color='#333', va='bottom', ha='right')

# Edit level annotations
edit_colors = {'L1': '#4caf50', 'L2': '#ff9800', 'L3': '#f44336', 'initial': '#9e9e9e'}
for it, sc, ed in zip(iters, scores, edits):
    c = edit_colors.get(ed, '#999')
    ax_a.annotate(ed, xy=(it, sc), xytext=(it, sc + 22),
                  fontsize=7.5, ha='center', color=c, fontweight='bold')

ax_a.set_xlabel('Iteration')
ax_a.set_ylabel('Development Score')
ax_a.set_title('(a) Development Score & Best-so-Far (seed_0, LunarLander-v3)',
               fontweight='bold')
ax_a.legend(loc='lower right')
ax_a.grid(axis='y', alpha=0.15)
ax_a.set_xlim(iters[0]-0.5, iters[-1]+0.5)

# --- (b) Activation rate heatmap ---
ax_b = fig.add_subplot(gs[1, 0])
act_matrix = np.zeros((len(all_comps), len(iters)))
for ci, comp in enumerate(all_comps):
    for ti, it in enumerate(iters):
        act_matrix[ci, ti] = iterations[it]['components'].get(comp, {}).get('nonzero_rate', 0)

im_b = ax_b.imshow(act_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax_b.set_xticks(range(len(iters)))
ax_b.set_xticklabels([str(i) for i in iters])
ax_b.set_yticks(range(len(all_comps)))
ax_b.set_yticklabels(all_comps, fontsize=7)
ax_b.set_title('(b) Component Activation Rate', fontweight='bold')
ax_b.set_xlabel('Iteration')
for ci in range(len(all_comps)):
    for ti in range(len(iters)):
        val = act_matrix[ci, ti]
        ax_b.text(ti, ci, f'{val:.2f}', ha='center', va='center',
                  fontsize=6.5, color='white' if val > 0.5 else '#333')
plt.colorbar(im_b, ax=ax_b, shrink=0.82)

# --- (c) Magnitude share heatmap ---
ax_c = fig.add_subplot(gs[1, 1])
mag_matrix = np.zeros((len(all_comps), len(iters)))
for ci, comp in enumerate(all_comps):
    for ti, it in enumerate(iters):
        comps_in_iter = iterations[it]['components']
        # magnitude share = abs(ep_sum_mean) / sum of all abs(ep_sum_mean)
        total_abs = sum(abs(comps_in_iter.get(c, {}).get('ep_sum_mean', 0))
                        for c in all_comps)
        if total_abs > 1e-12:
            mag_matrix[ci, ti] = abs(comps_in_iter.get(comp, {}).get('ep_sum_mean', 0)) / total_abs

im_c = ax_c.imshow(mag_matrix, aspect='auto', cmap='Blues', vmin=0, vmax=1)
ax_c.set_xticks(range(len(iters)))
ax_c.set_xticklabels([str(i) for i in iters])
ax_c.set_yticks(range(len(all_comps)))
ax_c.set_yticklabels(all_comps, fontsize=7)
ax_c.set_title('(c) Component Magnitude Share', fontweight='bold')
ax_c.set_xlabel('Iteration')
for ci in range(len(all_comps)):
    for ti in range(len(iters)):
        val = mag_matrix[ci, ti]
        ax_c.text(ti, ci, f'{val:.1%}', ha='center', va='center',
                  fontsize=6.5, color='white' if val > 0.5 else '#333')
plt.colorbar(im_c, ax=ax_c, shrink=0.82)

fig.suptitle('Fig.5: Reward Evolution Case Study — LunarLander seed_0 (Initial: −70.35 → Final: 224.21)',
             fontweight='bold', fontsize=11, y=1.01)

out = Path(__file__).resolve().parent / 'fig5_case_study.pdf'
fig.savefig(str(out))
print(f'Saved: {out}')
# Also save PNG for quick preview
png_out = Path(__file__).resolve().parent / 'fig5_case_study.png'
fig.savefig(str(png_out), dpi=150)
print(f'Saved: {png_out}')
plt.close()

# Print summary
print(f"\nIterations: {iters}")
print(f"Scores: {[f'{s:.1f}' for s in scores]}")
print(f"Edits: {edits}")
print(f"Components: {all_comps}")
