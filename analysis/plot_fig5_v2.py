"""Fig.5 v2: Case study triple with robust edit level parsing."""
import json, re, os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 8,
    'axes.labelsize': 9, 'axes.titlesize': 10,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

BASE = Path('c:/Users/Administrator/Downloads/expert_eureka_env001_bridge_v9_direct_generator/expert_eureka_env001_bridge_v9_direct_generator/runs/env_001/paper_v4/seed_0')
THRESHOLD = 200

# ── Extract edit level ─────────────────────────────────────────
def get_edit_level(reflection_path):
    if not os.path.exists(reflection_path):
        return 'initial' if 'iter_01' in str(reflection_path) else '?'
    t = open(reflection_path, 'r', encoding='utf-8').read()
    m = re.search(r'Level\s+(\d)', t)
    return f'L{m.group(1)}' if m else '?'

# ── Read iteration data ────────────────────────────────────────
iter_data = {}
for d in sorted(BASE.glob('iter_*')):
    it = int(d.name.split('_')[1])
    ts = d / 'training' / 'training_summary.json'
    es = d / 'training' / 'eval_result.json'
    rf = d / 'generation' / 'response_records' / 'agent_reflection.md'
    if not ts.exists() or not es.exists():
        continue

    train = json.loads(ts.read_text(encoding='utf-8'))
    ev = json.loads(es.read_text(encoding='utf-8'))

    comp_stats = train.get('component_summary', {}).get('component_stats', {})
    ep_comp = train.get('component_summary', {}).get('episode_component_sum_stats', {})

    edit_level = get_edit_level(str(rf))

    # Collect components with activation rate and magnitude share
    comps = {}
    total_abs = sum(abs(ep_comp.get(c.replace('component.', ''), {}).get('abs_mean', 0))
                    for c in comp_stats.keys() if 'component.' in c)
    for name, item in sorted(comp_stats.items()):
        short = name.replace('component.', '')
        if short in ('generated_reward', 'total_reward', 'original_env_reward'):
            continue
        ep_abs = abs(ep_comp.get(short, {}).get('abs_mean', 0))
        comps[short] = {
            'active_rate': item.get('nonzero_rate', 0),
            'mag_share': ep_abs / total_abs if total_abs > 1e-12 else 0,
        }

    iter_data[it] = {
        'score': ev['mean_eval_reward'],
        'edit': edit_level,
        'components': comps,
    }

iters = sorted(iter_data.keys())
scores = [iter_data[i]['score'] for i in iters]
edits = [iter_data[i]['edit'] for i in iters]
best_sofar = np.maximum.accumulate(scores)

# Collect all component names
all_comps = set()
for i in iters:
    all_comps.update(iter_data[i]['components'].keys())
all_comps = sorted(all_comps)

print(f"Iterations: {iters}")
print(f"Scores: {[f'{s:.1f}' for s in scores]}")
print(f"Edits: {edits}")
print(f"Components ({len(all_comps)}): {all_comps}")

# ── Plot ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.32, wspace=0.30)

# --- (a) Score trajectory ---
ax_a = fig.add_subplot(gs[0, :])
ax_a.plot(iters, scores, '-o', color='#1f77b4', linewidth=2, markersize=8,
          markerfacecolor='white', markeredgewidth=2, zorder=3, label='Per-iteration score')
ax_a.plot(iters, best_sofar, '--', color='#ff7f0e', linewidth=1.8, alpha=0.85,
          label='Best-so-far (Best Archive)')
ax_a.axhline(y=THRESHOLD, color='#333', linestyle='--', linewidth=1, alpha=0.5)
ax_a.annotate(f'Threshold={THRESHOLD}', xy=(iters[-1]+0.15, THRESHOLD),
             fontsize=7, color='#333', va='bottom', ha='right')

# Edit labels
edit_colors = {'L1': '#4caf50', 'L2': '#ff9800', 'L3': '#f44336', 'initial': '#9e9e9e', '?': '#bbb'}
for it, sc, ed in zip(iters, scores, edits):
    c = edit_colors.get(ed, '#999')
    offset = 30 if sc < 0 else 22
    ax_a.annotate(ed, xy=(it, sc), xytext=(it, sc + offset),
                  fontsize=7.5, ha='center', color=c, fontweight='bold')

ax_a.set_xlabel('Iteration')
ax_a.set_ylabel('Development Score')
ax_a.set_title('(a) Score & Best-so-Far (seed_0, LunarLander-v3)', fontweight='bold')
ax_a.legend(loc='lower right', framealpha=0.9)
ax_a.grid(axis='y', alpha=0.15)
ax_a.set_xlim(iters[0]-0.5, iters[-1]+0.5)

# --- (b) Activation rate ---
ax_b = fig.add_subplot(gs[1, 0])
act_matrix = np.zeros((len(all_comps), len(iters)))
for ci, comp in enumerate(all_comps):
    for ti, it in enumerate(iters):
        act_matrix[ci, ti] = iter_data[it]['components'].get(comp, {}).get('active_rate', 0)

im_b = ax_b.imshow(act_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax_b.set_xticks(range(len(iters)))
ax_b.set_xticklabels([str(i) for i in iters])
ax_b.set_yticks(range(len(all_comps)))
ax_b.set_yticklabels(all_comps, fontsize=6.5)
ax_b.set_title('(b) Component Activation Rate', fontweight='bold')
ax_b.set_xlabel('Iteration')
for ci in range(len(all_comps)):
    for ti in range(len(iters)):
        val = act_matrix[ci, ti]
        ax_b.text(ti, ci, f'{val:.2f}', ha='center', va='center',
                  fontsize=6.2, color='white' if val > 0.5 else '#333')
plt.colorbar(im_b, ax=ax_b, shrink=0.82)

# --- (c) Magnitude share ---
ax_c = fig.add_subplot(gs[1, 1])
mag_matrix = np.zeros((len(all_comps), len(iters)))
for ci, comp in enumerate(all_comps):
    for ti, it in enumerate(iters):
        mag_matrix[ci, ti] = iter_data[it]['components'].get(comp, {}).get('mag_share', 0)

im_c = ax_c.imshow(mag_matrix, aspect='auto', cmap='Blues', vmin=0, vmax=1)
ax_c.set_xticks(range(len(iters)))
ax_c.set_xticklabels([str(i) for i in iters])
ax_c.set_yticks(range(len(all_comps)))
ax_c.set_yticklabels(all_comps, fontsize=6.5)
ax_c.set_title('(c) Component Magnitude Share', fontweight='bold')
ax_c.set_xlabel('Iteration')
for ci in range(len(all_comps)):
    for ti in range(len(iters)):
        val = mag_matrix[ci, ti]
        ax_c.text(ti, ci, f'{val:.1%}', ha='center', va='center',
                  fontsize=6.2, color='white' if val > 0.5 else '#333')
plt.colorbar(im_c, ax=ax_c, shrink=0.82)

fig.suptitle('Reward Evolution Case Study — CREATE seed_0 (iter_01: -70.35 -> iter_08: 224.21)',
             fontweight='bold', fontsize=11, y=1.01)

out_dir = Path(__file__).resolve().parent
fig.savefig(str(out_dir / 'fig5_case_study.pdf'))
fig.savefig(str(out_dir / 'fig5_case_study.png'), dpi=150)
print(f'Saved: fig5_case_study.pdf and .png')
plt.close()
