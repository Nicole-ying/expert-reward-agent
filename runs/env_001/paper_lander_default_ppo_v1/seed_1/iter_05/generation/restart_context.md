# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -92.000

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| progress_reward + soft_landing_bonus + stability_penalty | 4 | -92.000 | -92.000 | unsolved |

## Previous interventions

- iter 2 (score=-106.270, structure=progress_reward + soft_landing_bonus + stability_penalty): 4. selected_level: Level 1**：三个组件的数学形态与职责符号均合理，问题明确是stability_penalty系数过强导致尺度失衡（|penalty/progress|=4.15，远超0.5的经验触发线）。 | 5. selected_intervention
- iter 3 (score=-107.990, structure=progress_reward + soft_landing_bonus + stability_penalty): `selected_level`: Level 2 — `sparse_to_dense` transformation of soft_landing_bonus, triggered by active_rate < 1% with all episodes crashing before landing. | `selected_intervention`: replace the hard 6-condition binary `soft_landing_bonus` with a continuous product of bounded proximity, speed, and angle scores. Each factor uses `max(0, 1 - value/threshold)` so partial improve
- iter 4 (score=-92.000, structure=progress_reward + soft_landing_bonus + stability_penalty): `selected_level`：Level 2 — 触发条件为`product_to_noncollapsing_joint`：三个[0,1]因子乘积导致奖励几乎恒塌缩至零，单因子改善无法产生有意义反馈。 | `selected_intervention`：将soft_landing_bonus从乘积 `proximity_score * speed_score * angle_score * 2.0` 改为几何平均 `(proximity_score * speed_score * angle_score) ** (1.0/3.0) * 2.0`。保持三个bounded因子和系数不变。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
