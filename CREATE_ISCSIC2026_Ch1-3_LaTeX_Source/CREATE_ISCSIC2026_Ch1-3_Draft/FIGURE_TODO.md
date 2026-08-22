# Final experiment-figure plan for the 6-page ISCSIC 2026 paper

All experiment figures must occupy **one IEEE column**, not two columns. Place each final vector PDF in `figures/` using the exact name below. Each figure is positioned inside the subsection whose claim it supports; do not combine the three figures into one multi-column summary.

## Common export requirements

- Final width: **3.35--3.45 in** (about one IEEE column).
- White background; vector PDF preferred.
- No figure title inside the image. The LaTeX caption supplies the title.
- Font after insertion: approximately **7.5--8 pt**, not smaller than 7 pt.
- Axis lines: about 0.8 pt; data lines: 1.2--1.5 pt; markers: 3.5--4.5 pt.
- Use grayscale-safe encodings: different line styles and marker shapes, not color alone.
- Do not place a large legend over data. Use a compact legend below or in an empty corner.
- Do not use 3-D effects, gradients, thick borders, or background grid boxes. Only light horizontal grid lines if needed.
- Use the unrounded JSON values for the final paper whenever available. Rounded values below are for layout and checking.

---

## Fig. 2: `fig2_budget_curve.pdf`

### Corresponding subsection

`IV-B Reward Search Under a Matched Training Budget`

The LaTeX placeholder is already located directly after the main budget-comparison discussion.

### Exact chart type

**Single-panel line chart**. Do not use bars, box plots, or a three-panel composite.

### Intended claim

Show whether additional policy-training evaluations are converted into cumulative reward-search progress. The table already reports final means and solved counts, so this figure should show the **search process**, not repeat endpoint statistics.

### Axes

- x-axis: `Reward evaluations`, integer ticks 1--10.
- y-axis: `Best-so-far search fitness`.
- Suggested y-range: approximately -130 to 270.
- Add one horizontal dashed line at 200 labeled `Task threshold`.

### Curves

Plot exactly three mean curves:

1. `Independent-10`
2. `Stateless revision`
3. `CREATE`

For method $m$, seed $s$, and evaluation $t$, first calculate

`best_so_far[m,s,t] = max(fitness[m,s,1:t])`.

Then average the five lineage-wise best-so-far values at every budget. For CREATE lineages stopped early, carry the archived best value forward to budget 10. For Independent-10, columns c0--c9 are treated as evaluation budgets 1--10; their ordering does not imply revision.

### Visual encoding

- One bold mean line per method.
- Use a different marker and line style for each method.
- Do **not** add three overlapping uncertainty bands in a one-column plot.
- Do **not** plot all 15 seed traces; they will make the figure unreadable.
- Variability is reported in Table III and Fig. 3.

### Caption already prepared in LaTeX

`Best-so-far LunarLander search fitness under a matched policy-training budget...`

### Data already available

- Independent-10: complete 5 x 10 candidate matrix supplied.
- Stateless revision: complete 5 x 10 iteration matrix supplied.
- CREATE: complete five-lineage iteration matrix supplied.

### Still required for the final rendering

Export the same matrices with the **full floating-point search fitness**, not only rounded integers. Recommended CSV format:

```text
method,seed,evaluation,current_fitness
Independent-10,0,1,...
Stateless,0,1,...
CREATE,0,1,...
```

Also include a boolean `stopped` or use blank values after stopping; the plotting script should carry the incumbent forward only for best-so-far computation.

---

## Fig. 3: `fig3_ablation_scatter.pdf`

### Corresponding subsection

`IV-C Mechanism Ablations`

The LaTeX placeholder is placed after the two mechanism-ablation paragraphs.

### Exact chart type

**Categorical jittered dot plot (strip plot) with mean and sample-standard-deviation marker**.

Do not use a box plot: each condition has only five lineages, and a box plot would hide the actual observations. Do not use a bar chart: the evidence-enrichment ablation contains both successful and strongly unsuccessful seeds, so a mean bar is misleading.

### Intended claim

Show that the complete system improves both task-solving rate and cross-seed reliability, while the two damaged variants fail in different ways.

### Axes and groups

- x-axis: three categories only:
  1. `w/o Evidence`
  2. `w/o Hierarchy`
  3. `CREATE`
- y-axis: `Best search fitness`.
- Suggested y-range: -150 to 280.
- Horizontal dashed threshold at 200.

### Points

Use all five lineage-wise best values in each group:

- `w/o Evidence`: 240, 170, -110, 116, 260
- `w/o Hierarchy`: 170, 131, 71, 59, 140
- `CREATE`: use the unrounded values 224.21, 240.60, 220.24, 253.71, 206.14

Apply only slight horizontal jitter so that overlapping points remain visible.

### Summary markers

- Add a short horizontal mean line per group.
- Add a thin vertical sample-standard-deviation whisker if it remains legible.
- Add compact solved labels near the group top or beneath the x labels: `2/5`, `0/5`, and `5/5`.
- Do not connect points across groups because the runs are not guaranteed to be paired identical initial rewards.

### Still required for the final rendering

Provide unrounded best-fitness values for both ablations if available. The integer values above can be used only as a temporary draft.

Recommended CSV:

```text
condition,seed,best_search_fitness,solved
w_o_evidence,0,...,1
w_o_hierarchy,0,...,0
CREATE,0,...,1
```

---

## Fig. 4: `fig4_repair_trace.pdf`

### Corresponding subsection

`IV-D Evidence-to-Edit Case Study`

This figure is reserved only for the CREATE seed-0 decisive transition from round 7 to round 8. It should not contain ablation statistics or BipedalWalker results.

### Exact figure type

**One-column case-study process figure**, not a conventional statistical chart. Use a vertical layout with two tightly related parts:

1. A compact line plot of the seed-0 reward-search trace.
2. A verified evidence--diagnosis--action--outcome chain for the round-7 to round-8 edit.

These two parts belong to the same case-study subsection and may be vertically aligned inside one figure. Do not add a third unrelated panel.

### Upper part: compact line plot

Plot two lines over reward-design rounds:

- `Current fitness`
- `Archived best fitness`

Requirements:

- x-axis: `Reward-design round`.
- y-axis: `Search fitness`.
- Show rounds 1--9 for seed 0.
- Add the 200 threshold.
- Mark round 7, round 8, and round 9.
- Annotate the decisive transition `L2 repair` between rounds 7 and 8.
- The plot should show the round-9 regression while the archive remains at the solved round-8 value.

Use exact floating-point fitness from the run log. Rounded draft values are approximately:

`-70, -129, -26, -120, -124, -18, -389, 224, -612`.

### Lower part: evidence-to-edit chain

Use four compact boxes connected vertically or horizontally:

1. `Measured evidence at round 7`
2. `Reward-editor diagnosis`
3. `Selected semantic and L2 code change`
4. `Observed round-8 outcome`

Each box should contain at most 2--3 short lines. The auxiliary subagent belongs only in box 1 as the organizer of measured evidence; it must not be shown as selecting the edit.

### Information required from you before this figure and subsection can be finalized

#### A. Round-level evaluation summary

For CREATE seed 0, rounds 7, 8, and preferably 9:

```text
round
search_fitness (full precision)
mean_episode_length
terminated_count / 20
truncated_count / 20
success_count, crash_count, timeout_count, or the exact available termination categories
```

Do not infer success from `terminated_count`; provide the actual outcome categories used by the evaluator.

#### B. Component evidence before and after the decisive edit

For round 7 and round 8, provide one row per reward component:

```text
component_name
episode_sum_mean
active_rate
magnitude_share
signed_contribution_share, if available
early_training_value
mid_training_value
late_training_value
delta_from_previous_round, if available
```

From the full list, the final figure will retain only 3--5 components:

- the component implementing the edited semantic;
- one downstream task-event component affected by that semantic;
- one competing or dominating component, if present;
- optionally one unchanged reference component.

Do not preselect components only because their values look dramatic. Selection must follow the actual reward-editor diagnosis.

#### C. Reward-editor decision record

Provide the exact structured output for the round-7 decision:

```text
principal_failure_or_hypothesis
selected_semantic q_t
edit_level L1/L2/L3
selected_component_or_code_region
reason for choosing this semantic
expected next-run change
main counter-risk
```

The paper currently states that this was an L2 refactor; the saved decision record is needed to verify that wording.

#### D. Reward-code difference

Provide either the two full reward functions or a clean diff containing:

```text
round-7 expression before edit
round-8 expression after edit
weights/gates/conditions added, removed, or changed
components intentionally left unchanged
```

The main paper will show only a one- or two-line mathematical/code summary. Full code belongs in the supplement.

#### E. Outcome verification

For every prediction made before round 8, report whether the round-8 evidence supported it. A compact format is:

```text
prediction,observed_change,supported_or_not
```

This is necessary to write a mechanism analysis rather than merely state that fitness increased.

### What will be written from these materials

The final subsection will explicitly separate:

- what the evidence organizer measured and compressed;
- what failure the Reward Editing Agent diagnosed;
- why L2 rather than L1 or L3 was selected;
- what code semantic changed;
- whether the predicted component/behavioral change occurred;
- how the best archive protected the solution after the later regression.

If active-rate and magnitude-share records are unavailable, the figure must be simplified to the fitness/archive timeline plus the verified textual decision chain. The paper must not infer component behavior from fitness alone.

---

## BipedalWalker-v3: table only, no main-paper figure

### Corresponding subsection

`IV-E Additional Validation on Continuous Control`

The main paper already contains a one-column table with:

- initial fitness;
- first reward version reaching 300;
- best fitness;
- five-seed mean and sample standard deviation.

Do not draw another BipedalWalker line chart in the six-page paper. The five complete version trajectories should be placed in the supplementary material. The main text should analyze only:

- all five lineages reaching the threshold;
- four near-threshold initial rewards requiring local refinement;
- seed 4 as the more informative difficult initialization;
- the limitation that no BipedalWalker baseline or ablation was run.

---

## Final figure-to-subsection mapping

| Figure/table | Type | Paper subsection | Main purpose |
|---|---|---|---|
| Table III | Numeric result table | IV-A / IV-B | Final baseline and ablation outcomes |
| Fig. 2 | One-column line chart | IV-B | Budget-dependent cumulative search progress |
| Fig. 3 | One-column jittered dot plot | IV-C | Mechanism reliability across five seeds |
| Fig. 4 | One-column case-study process figure | IV-D | Evidence-to-diagnosis-to-edit chain |
| Bipedal table | One-column numeric table | IV-E | Additional continuous-control validation |

This layout prevents figures from being detached from their discussion and avoids a single two-column composite that tries to answer several unrelated experimental questions.
