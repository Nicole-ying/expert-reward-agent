# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -108.370

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| distance_reward + landing_quality + stability_penalty | 3 | -108.370 | -109.790 | unsolved |
| distance_reward + soft_landing_bonus + stability_penalty | 1 | -110.970 | -110.970 | unsolved |

## Previous interventions

- iter 2 (score=-112.960, structure=distance_reward + landing_quality + stability_penalty): 4. selected_level: Level 2** — the 0.5% active_rate binary bonus matches the `sparse_to_dense` evidence pattern exactly; a coefficient tweak cannot fix structural sparsity. | 5. selected_intervention
- iter 4 (score=-109.790, structure=distance_reward + landing_quality + stability_penalty): selected_level**：Level 1 —— 稳定性惩罚的职责和数学形态合理，但尺度过强阻碍早期探索和必要机动。上一轮修改了landing_quality形态（已生效），本轮不应重复修改同一组件；stability_penalty系数尚未被调整过，且ratio虽低于0.5但处于crash场景下的"过度约束"状态。 | selected_intervention**：仅调整stability_penalty的四个权重系数，将其整体幅度降至约40%（w_vx: 0.15→0.06, w_vy: 0.05→0.02, w_angle: 0.2→0.08, w_angvel: 0.2→0.08）。distance_reward和landing_quality保持不变。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
