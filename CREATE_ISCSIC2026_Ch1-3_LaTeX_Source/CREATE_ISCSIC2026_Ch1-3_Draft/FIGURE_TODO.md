# Experiment figure plan for the 6-page ISCSIC 2026 paper

The LaTeX source already reserves the required space. Place the final PDFs in `figures/` using the exact names below.

## 1. `fig2_main_results.pdf` — required, full width

Recommended physical size: about 7.0–7.15 in wide and 2.0–2.25 in high. Use vector PDF, white background, consistent IEEE-size text, and no title inside the figure.

Three panels:

### (a) Best-so-far search fitness vs. reward-evaluation budget
- x-axis: reward evaluations, 1–10.
- y-axis: best-so-far search fitness.
- Plot `Independent-10`, `Stateless Iterative Revision`, and `CREATE`.
- For each seed use `B[s,t] = max_{k<=t} fitness[s,k]`; carry the incumbent to budget 10 after early stopping.
- Thick line: mean over five seeds.
- Light band: min–max over five seeds; do not label it as a confidence interval.
- Horizontal dashed threshold: 200.

### (b) Success@Budget
- x-axis: reward evaluations, 1–10.
- y-axis: fraction solved, 0–1.
- Determine success separately for every seed before averaging.
- CREATE values: `0, .2, .4, .6, .6, .6, .6, .8, .8, 1.0`.
- The three baselines remain at 0.
- Use a step plot.

### (c) Five-seed best-fitness scatter
Six groups:
1. Single-shot Generation
2. Independent-10
3. Stateless Iterative Revision
4. CREATE w/o Evidence Enrichment
5. CREATE w/o Hierarchical Semantic Editing
6. CREATE

- Show all five points per group.
- Add a short horizontal mean marker.
- Add the 200 threshold line.
- Do not use bars; the evidence-enrichment ablation has very high variance.
- Short x labels are acceptable: `Single`, `Indep.-10`, `Stateless`, `w/o Evidence`, `w/o Hierarchy`, `CREATE`.

Do **not** reuse the old `success_by_budget.pdf`: its earlier script computed the independent baseline success from a bootstrapped mean rather than the fraction of solved seeds.

## 2. `fig3_repair_case.pdf` — required final replacement, one column

Recommended size: about 3.35–3.45 in wide and 2.4–2.8 in high. The source temporarily falls back to the old `ISCSIC2026_framework/paper/figures/repair_case.pdf` until this file is added.

Suggested panels:

### (a) Seed-0 reward-search trace
- Current search fitness by round.
- Best-so-far/archive fitness by round.
- Threshold 200.
- Mark the decisive round-7 to round-8 repair and the later failed exploration.

### (b) Decisive component evidence, round 7 vs. round 8
- Only 4–6 components relevant to the actual edit.
- Show active rate and magnitude share; two compact aligned plots are preferable to a large heatmap.
- Use the exact component names and values from the saved logs.

### (c) Evidence–diagnosis–action–outcome chain
Use five compact boxes:
`Training evidence -> Failure diagnosis -> Selected semantic -> L2 refactor -> Fitness 224`.
The diagnostic and component claims must come from the actual reward-editor record, not from the fitness curve alone.

## Existing old figures

- `performance_curve.pdf`: may be used for internal comparison, but it does not cover the finalized baselines and should not replace `fig2_main_results.pdf`.
- `ablation.pdf`: labels and visual density do not match the finalized ablation definitions; redraw panel (c) instead.
- `repair_case.pdf`: retained only as a temporary fallback until `fig3_repair_case.pdf` is supplied.
- `independent_test_scatter.pdf`: can be moved to supplementary material after all methods have a consistent 100-episode test evaluation.

## Material recommended for the supplementary PDF

- Complete PPO and LLM configuration.
- Exact prompts and output schema.
- Full 5 x 10 per-seed search matrices for every condition.
- All 100-episode test-fitness distributions.
- Full reward code before/after the representative repair.
- Complete component statistics and intervention-memory record.
- LLM calls, tokens, wall-clock time, environment steps, and peak candidate concurrency.
