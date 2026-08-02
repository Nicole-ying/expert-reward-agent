"""Supplementary figures for CREATE paper.
Outputs:
  held_out_vs_dev.pdf    — Held-out vs Dev scatter (all experiments)
  termination_evolution.pdf — Terminated ratio per iteration (seed_0)
  convergence_rounds.pdf — Rounds needed to reach threshold per seed
  edit_distribution.pdf  — L1/L2/L3 distribution across all seeds
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
HELD_OUT_DIR = OUT / 'held_out_eval'
THRESHOLD = 200

def save(fig, name):
    fig.savefig(str(OUT / f'{name}.pdf'))
    fig.savefig(str(OUT / f'{name}.png'), dpi=150)
    print(f'  {name}.pdf + .png')

# ====================================================================
# 1. Held-out vs Dev scatter — all experiments on one plot
# ====================================================================
def make_held_out_vs_dev():
    # Load held-out data
    held = {}
    for fname, label, color in [
        ('held_out_CREATE.json', 'CREATE', '#4CAF50'),
        ('held_out_CoarseFeedback.json', 'Coarse Feedback', '#FF9800'),
        ('held_out_Unconstrained.json', 'Unconstrained', '#f44336'),
        ('held_out_IndependentGen.json', 'Independent Gen', '#FF5722'),
    ]:
        d = json.loads((HELD_OUT_DIR / fname).read_text('utf-8'))
        held[label] = {'scores': [], 'dev': [], 'color': color}
        for seed_name, v in d.items():
            held[label]['scores'].append(v['held_out_mean'])
            held[label]['dev'].append(v.get('dev_score', None))

    # The dev scores for CREATE come from paper_v4, for ablations from their dirs
    # For consistency, use the held-out json dev_score field (already matches)
    # For Independent Gen, we need dev from the best candidates
    # Actually held_out json already has dev_score, let's use that

    fig, ax = plt.subplots(figsize=(6, 5.5))

    for label, data in held.items():
        devs = data['dev']
        hs = data['scores']
        c = data['color']
        ax.scatter(devs, hs, c=c, s=100, zorder=3, edgecolors='white',
                   linewidths=0.8, label=f"{label} ({mean(hs):.0f})")

    # Diagonal: dev = held_out
    lims = [-200, 350]
    ax.plot(lims, lims, '--', color='#999', linewidth=1, alpha=0.6, label='dev = held-out')

    # Threshold lines
    ax.axhline(y=THRESHOLD, color='#333', linestyle=':', linewidth=1, alpha=0.4)
    ax.axvline(x=THRESHOLD, color='#333', linestyle=':', linewidth=1, alpha=0.4)

    # Quadrant annotation
    ax.annotate('Both > threshold\n(good + generalizes)', xy=(280, 280),
                fontsize=7.5, color='#4CAF50', ha='center', alpha=0.7)
    ax.annotate('dev < threshold\nheld-out > threshold\n(undetected good)', xy=(50, 270),
                fontsize=7.5, color='#999', ha='center', alpha=0.6)
    ax.annotate('dev > threshold\nheld-out < threshold\n(overfit to dev)', xy=(280, 50),
                fontsize=7.5, color='#999', ha='center', alpha=0.6)

    ax.set_xlabel('Development Score')
    ax.set_ylabel('Held-out Score (100 episodes)')
    ax.set_title('Held-out vs Development Score (all methods, all seeds)',
                 fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9, fontsize=8)
    ax.set_xlim(-180, 330)
    ax.set_ylim(-180, 330)
    ax.grid(alpha=0.12)

    save(fig, 'held_out_vs_dev')


# ====================================================================
# 2. Termination mode evolution — seed_0 across iterations
# ====================================================================
def make_termination_evolution():
    seed_dir = BASE / 'paper_v4' / 'seed_0'
    iters, term_rates = [], []

    for d in sorted(seed_dir.glob('iter_*')):
        it = int(d.name.split('_')[1])
        ef = d / 'training' / 'eval_result.json'
        if not ef.exists():
            continue
        ev = json.loads(ef.read_text(encoding='utf-8'))
        terms = ev.get('episode_terminated', [])
        if terms:
            iters.append(it)
            term_rates.append(sum(1 for t in terms if t) / len(terms))

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    colors = ['#4CAF50' if r >= 0.8 else '#FF9800' if r >= 0.4 else '#f44336'
              for r in term_rates]
    bars = ax.bar(iters, term_rates, color=colors, edgecolor='white', linewidth=0.5)

    for it, r in zip(iters, term_rates):
        ax.annotate(f'{r:.0%}', xy=(it, r), xytext=(it, r + 0.04),
                    fontsize=9, ha='center', fontweight='bold')

    # Annotate threshold-crossing
    best_iter = iters[np.argmax([json.loads(
        (seed_dir / f'iter_{i:02d}' / 'training' / 'eval_result.json').read_text('utf-8')
    )['mean_eval_reward'] for i in iters])]
    ax.annotate('Best Archive\n(score >= 200)', xy=(best_iter, term_rates[iters.index(best_iter)]),
                xytext=(best_iter + 1.5, term_rates[iters.index(best_iter)] - 0.2),
                fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', color='#333', alpha=0.6))

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Terminated Rate (not timeout)')
    ax.set_title('Termination Evolution — CREATE seed_0 (LunarLander-v3)',
                 fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_xticks(iters)
    ax.grid(axis='y', alpha=0.15)

    save(fig, 'termination_evolution')


# ====================================================================
# 3. Convergence rounds — how many iterations to reach threshold
# ====================================================================
def make_convergence_rounds():
    convergence = {}  # seed -> first iteration where score >= 200
    for seed_dir in sorted((BASE / 'paper_v4').glob('seed_*')):
        seed = int(seed_dir.name.split('_')[1])
        for d in sorted(seed_dir.glob('iter_*')):
            it = int(d.name.split('_')[1])
            ef = d / 'training' / 'eval_result.json'
            if ef.exists():
                score = json.loads(ef.read_text(encoding='utf-8'))['mean_eval_reward']
                if score >= THRESHOLD and seed not in convergence:
                    convergence[seed] = it

    # Also check Coarse Feedback
    coarse_conv = {}
    for seed_dir in sorted((BASE / 'ablation_eureka_feedback_v4').glob('seed_*')):
        seed = int(seed_dir.name.split('_')[1])
        for d in sorted(seed_dir.glob('iter_*')):
            it = int(d.name.split('_')[1])
            ef = d / 'training' / 'eval_result.json'
            if ef.exists():
                score = json.loads(ef.read_text(encoding='utf-8'))['mean_eval_reward']
                if score >= THRESHOLD and seed not in coarse_conv:
                    coarse_conv[seed] = it

    fig, ax = plt.subplots(figsize=(6, 3.5))

    x = np.arange(5)
    width = 0.35
    create_rounds = [convergence.get(s, 11) for s in range(5)]  # 11 = never solved
    coarse_rounds = [coarse_conv.get(s, 11) for s in range(5)]

    bars1 = ax.bar(x - width/2, create_rounds, width, color='#4CAF50',
                   edgecolor='white', label='CREATE')
    bars2 = ax.bar(x + width/2, coarse_rounds, width, color='#FF9800',
                   edgecolor='white', label='Coarse Feedback')

    # Label each bar
    for bar, val in zip(bars1, create_rounds):
        label = str(val) if val <= 10 else 'N/A'
        ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    fontsize=9, ha='center', fontweight='bold', color='#4CAF50')
    for bar, val in zip(bars2, coarse_rounds):
        label = str(val) if val <= 10 else 'N/A'
        ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    fontsize=9, ha='center', fontweight='bold', color='#FF9800')

    ax.axhline(y=10, color='#999', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.annotate('Max budget', xy=(4.5, 10), fontsize=7, color='#999', ha='left', va='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels([f'seed_{s}' for s in range(5)])
    ax.set_ylabel('Iterations to Threshold (200)')
    ax.set_title('Convergence Speed — CREATE vs Coarse Feedback', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 13)
    ax.grid(axis='y', alpha=0.15)

    save(fig, 'convergence_rounds')


# ====================================================================
# 4. Edit level distribution (L1/L2/L3 across all CREATE seeds)
# ====================================================================
def make_edit_distribution():
    edits = {'L1': 0, 'L2': 0, 'L3': 0, '?': 0}
    per_seed = {}

    for seed_dir in sorted((BASE / 'paper_v4').glob('seed_*')):
        seed = int(seed_dir.name.split('_')[1])
        seed_edits = {'L1': 0, 'L2': 0, 'L3': 0}
        for d in sorted(seed_dir.glob('iter_*')):
            it = int(d.name.split('_')[1])
            if it == 1: continue  # iter_01 is initial generation, not an edit
            rf = d / 'generation' / 'response_records' / 'agent_reflection.md'
            level = '?'
            if rf.exists():
                t = rf.read_text(encoding='utf-8')
                m = re.search(r'Level\s+(\d)', t)
                if m: level = f'L{m.group(1)}'
            edits[level] = edits.get(level, 0) + 1
            if level in seed_edits:
                seed_edits[level] += 1
        per_seed[seed] = seed_edits

    # Pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    labels = ['L1: Parameter Tuning', 'L2: Component Refactoring', 'L3: Full Redesign']
    sizes = [edits['L1'], edits['L2'], edits['L3']]
    colors_pie = ['#4CAF50', '#FF9800', '#f44336']
    explode = (0.02, 0.02, 0.08)

    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=labels, colors=colors_pie,
        autopct='%1.1f%%', startangle=140,
        textprops={'fontsize': 9})
    for at in autotexts:
        at.set_fontweight('bold')
    ax1.set_title('Edit Level Distribution (all seeds)', fontweight='bold')

    # Per-seed breakdown
    seeds = sorted(per_seed.keys())
    x = np.arange(len(seeds))
    width = 0.25
    l1_vals = [per_seed[s]['L1'] for s in seeds]
    l2_vals = [per_seed[s]['L2'] for s in seeds]
    l3_vals = [per_seed[s]['L3'] for s in seeds]

    ax2.bar(x - width, l1_vals, width, color='#4CAF50', edgecolor='white', label='L1')
    ax2.bar(x, l2_vals, width, color='#FF9800', edgecolor='white', label='L2')
    ax2.bar(x + width, l3_vals, width, color='#f44336', edgecolor='white', label='L3')

    # Annotate
    for i in range(len(seeds)):
        total = l1_vals[i] + l2_vals[i] + l3_vals[i]
        ax2.annotate(str(total), xy=(i + width, max(l1_vals[i], l2_vals[i], l3_vals[i])),
                     xytext=(0, 8), textcoords='offset points', fontsize=8, ha='center',
                     fontweight='bold', color='#333')

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'seed_{s}' for s in seeds])
    ax2.set_ylabel('Number of Edits')
    ax2.set_title('Edits per Seed', fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax2.grid(axis='y', alpha=0.15)

    fig.suptitle('Edit Level Analysis — CREATE (LunarLander-v3, 5 seeds, 35 non-initial iterations)',
                 fontweight='bold', fontsize=11, y=1.02)

    save(fig, 'edit_distribution')


# ====================================================================
if __name__ == '__main__':
    print("Generating supplementary figures...")
    make_held_out_vs_dev()
    make_termination_evolution()
    make_convergence_rounds()
    make_edit_distribution()
    print("\nDone.")
