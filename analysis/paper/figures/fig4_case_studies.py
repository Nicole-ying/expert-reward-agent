"""Fig 4: Repair case studies — adapted from plot_fig5_case_study.py"""
import json, re, numpy as np, matplotlib.pyplot as plt, matplotlib
from pathlib import Path
from collections import OrderedDict

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 8,
    'axes.labelsize': 9, 'axes.titlesize': 10,
    'legend.fontsize': 7, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

BASE = Path('/home/utseus22/expert-reward-agent/runs/env_001/paper_v4')
THRESHOLD = 200

def load_iteration_data(seed, it):
    """Load score + components for one iteration."""
    ts = BASE / f'seed_{seed}' / f'iter_{it:02d}' / 'training' / 'training_summary.json'
    es = BASE / f'seed_{seed}' / f'iter_{it:02d}' / 'training' / 'eval_result.json'
    if not ts.exists() or not es.exists():
        return None
    train = json.loads(ts.read_text(encoding='utf-8'))
    ev = json.loads(es.read_text(encoding='utf-8'))
    comp_stats = train.get('component_summary', {}).get('component_stats', {})
    ep_comp = train.get('component_summary', {}).get('episode_component_sum_stats', {})
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
    return {
        'score': ev['mean_eval_reward'],
        'score_min': ev.get('min_eval_reward', ev['mean_eval_reward']),
        'score_max': ev.get('max_eval_reward', ev['mean_eval_reward']),
        'components': comps,
    }

def get_edit_level(seed, it):
    """Classify by actual intervention content, not LLM self-declaration."""
    if it == 1:
        return 'initial'
    resp = BASE / f'seed_{seed}' / f'iter_{it:02d}' / 'generation' / 'response_records' / 'agent_reflection.md'
    if not resp.exists():
        return 'L2'
    text = resp.read_text(encoding='utf-8')
    # Bare-code = validation retry, inherit previous
    if not re.search(r'selected_level|evidence|behavior_diagnosis|selected_intervention', text):
        return get_edit_level(seed, it - 1)
    # Extract intervention description
    m = re.search(r'(?:selected_intervention|intervention).*?[：:]\s*(.+?)(?=\n\d+\.|\n```|\n\Z)', text, re.DOTALL)
    intervention = (m.group(1) if m else text)[:300]

    # L3: skeleton rebuild
    if any(w in intervention for w in ['Level 3', 'Level3', '重建', 'new skeleton', 'restart']):
        return 'L3'
    # L2: formula form change, structural change
    l2_kw = ['替换', 'replace', '稀疏', 'continuous', '改为', 'delta', 'remove', '移除', '去掉',
             '加入', '新增', '拆分', 'split', 'transition', 'sustained', 'state-based', 'product',
             'gated', 'changed from', 'instead of', 'morphology', '重整', '重写', 'rewrite']
    if any(w in intervention for w in l2_kw):
        return 'L2'
    # L1: coefficient/weight only
    l1_kw = ['系数', 'coefficient', '权重', 'weight', '提高到', '降低到', '降至', '升至',
             'reduce', 'increase', 'scale', 'tune', '调']
    if any(w in intervention for w in l1_kw):
        return 'L1'
    return 'L2'

edit_colors = {'L1': '#4caf50', 'L2': '#ff9800', 'L3': '#f44336', 'initial': '#9e9e9e'}

def make_score_figure(seed, max_iter, fname):
    """Single seed score trajectory with edit-level annotations."""
    iters_data = {}
    for it in range(1, max_iter + 1):
        d = load_iteration_data(seed, it)
        if d:
            iters_data[it] = d
    if not iters_data:
        return
    iters = sorted(iters_data.keys())
    scores = [iters_data[i]['score'] for i in iters]
    edits = [get_edit_level(seed, i) for i in iters]

    fig, ax = plt.subplots(figsize=(3.45, 2.0))

    # Search fitness — thin line + solid markers
    ax.plot(iters, scores, '-o', color='#1f77b4', linewidth=0.8, markersize=4,
            markerfacecolor='#1f77b4', markeredgewidth=0, zorder=5,
            label='Search fitness')

    # Best search fitness
    best_sofar = list(np.maximum.accumulate(scores))
    ax.plot(iters, best_sofar, '--', color='#ff7f0e', linewidth=0.8, alpha=0.9,
            label='Best search fitness', zorder=6)

    # Threshold (line only, no label)
    ax.axhline(y=THRESHOLD, color='#333', linestyle='--', linewidth=0.6, alpha=0.35)

    # L1/L2/L3 labels — below the line, per-seed manual tweaks
    y_min, y_max = min(scores), max(scores)
    y_range = max(y_max - y_min, 50)
    # (seed, iter) -> (x_off, y_off) manual adjustments
    tweaks = {
        (0, 8): (0.10, 0), (0, 6): (-0.10, 0),
        (3, 4): (0.05, 0), (3, 2): (0, -10),
    }
    for it, sc, ed in zip(iters, scores, edits):
        if ed == 'initial':
            continue
        c = edit_colors.get(ed, '#999')
        dx, dy = tweaks.get((seed, it), (0, 0))
        ax.text(it + dx, sc - y_range * 0.03 - 2 + dy, ed, fontsize=5.5,
                ha='center', va='top', color=c, fontweight='bold', clip_on=True)

    ax.set_xlabel('Reward-design round')
    ax.set_ylabel('Search fitness')
    ax.legend(loc='upper left', framealpha=0.85)
    ax.grid(axis='y', alpha=0.12)
    ax.set_xticks(iters)
    ax.set_xticklabels([str(i) for i in iters])
    ax.set_ylim(y_min - y_range * 0.18, y_max + y_range * 0.18)
    ax.set_xlim(iters[0] - 0.5, iters[-1] + 0.5)
    for sp in ax.spines.values(): sp.set_linewidth(0.5)

    out = Path(__file__).resolve().parent / 'figures' / fname
    fig.savefig(str(out.with_suffix('.pdf')))
    fig.savefig(str(out.with_suffix('.png')), dpi=150)
    print(f'Saved: {out}.pdf (seed {seed}, iters {iters[0]}-{iters[-1]})')
    plt.close(fig)

# ── Seed 0: rounds 1-8 (solves at r8) ──
make_score_figure(0, 8, 'fig4a_seed0_repair_trace')

# ── Seed 3: rounds 1-4 (solves at r4) ──
make_score_figure(3, 4, 'fig4b_seed3_repair_trace')

# ── Seed 4: rounds 1-3 (solves at r3) ──
make_score_figure(4, 3, 'fig4c_seed4_repair_trace')

# ── Supplementary: Component heatmaps for all three seeds ──
for seed, max_r, tag in [(0, 8, 'seed0'), (3, 4, 'seed3'), (4, 3, 'seed4')]:
    print(f"\n--- Heatmaps: Seed {seed} (rounds 1-{max_r}) ---")
    iters_data = {}
    all_comps = set()
    for it in range(1, max_r + 1):
        d = load_iteration_data(seed, it)
        if d:
            iters_data[it] = d
            all_comps.update(d['components'].keys())
    all_comps = sorted(all_comps)
    iters = sorted(iters_data.keys())

    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(8, 3.0))

    # (b) Activation rate heatmap
    act_matrix = np.zeros((len(all_comps), len(iters)))
    for ci, comp in enumerate(all_comps):
        for ti, it in enumerate(iters):
            act_matrix[ci, ti] = iters_data[it]['components'].get(comp, {}).get('nonzero_rate', 0)
    im_b = ax_b.imshow(act_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax_b.set_xticks(range(len(iters))); ax_b.set_xticklabels([str(i) for i in iters])
    ax_b.set_yticks(range(len(all_comps))); ax_b.set_yticklabels(all_comps, fontsize=7)
    for ci in range(len(all_comps)):
        for ti in range(len(iters)):
            val = act_matrix[ci, ti]
            ax_b.text(ti, ci, f'{val:.2f}', ha='center', va='center',
                      fontsize=6, color='white' if val > 0.5 else '#333')
    plt.colorbar(im_b, ax=ax_b, shrink=0.85)
    ax_b.set_xlabel('Iteration')

    # (c) Magnitude share heatmap
    mag_matrix = np.zeros((len(all_comps), len(iters)))
    for ci, comp in enumerate(all_comps):
        for ti, it in enumerate(iters):
            comps_in_iter = iters_data[it]['components']
            total_abs = sum(abs(comps_in_iter.get(c, {}).get('ep_sum_mean', 0)) for c in all_comps)
            if total_abs > 1e-12:
                mag_matrix[ci, ti] = abs(comps_in_iter.get(comp, {}).get('ep_sum_mean', 0)) / total_abs
    im_c = ax_c.imshow(mag_matrix, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax_c.set_xticks(range(len(iters))); ax_c.set_xticklabels([str(i) for i in iters])
    ax_c.set_yticks(range(len(all_comps))); ax_c.set_yticklabels(all_comps, fontsize=7)
    for ci in range(len(all_comps)):
        for ti in range(len(iters)):
            val = mag_matrix[ci, ti]
            ax_c.text(ti, ci, f'{val:.1%}', ha='center', va='center',
                      fontsize=6, color='white' if val > 0.5 else '#333')
    plt.colorbar(im_c, ax=ax_c, shrink=0.85)
    ax_c.set_xlabel('Iteration')

    out = Path(__file__).resolve().parent / 'figures' / f'figS1_{tag}_heatmaps'
    fig.savefig(str(out.with_suffix('.pdf')))
    fig.savefig(str(out.with_suffix('.png')), dpi=150)
    print(f'Saved: {out}.pdf')
    plt.close(fig)
