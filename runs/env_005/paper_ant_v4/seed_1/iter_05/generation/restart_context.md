# Fresh Restart Evidence

- target_score: 2000.000
- best_score_so_far: 434.480

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| forward_progress + height_penalty + lateral_drift_penalty + upright_stability | 4 | 434.480 | 86.940 | unsolved |

## Previous interventions

- iter 2 (score=-224.850, structure=forward_progress + height_penalty + lateral_drift_penalty + upright_stability): 4. **selected_level**：Level 1——尺度修复。forward_progress 的系数过小导致其被 upright 信号完全主导，只需增大其系数即可改变相对重要性。 | 5. **selected_intervention**：仅将 forward_progress 的倍率从 2.0 改为 20.0，其他所有组件、结构、约束保持不变。
- iter 3 (score=265.020, structure=forward_progress + height_penalty + lateral_drift_penalty + upright_stability): 4. **selected_level**：Level 1，因forward_progress数学形态合理但相对upright_stability的尺度不足，只需调整其系数，不改变结构。 | 5. **selected_intervention**：将forward_progress的缩放系数从20.0提高至60.0，其他所有组件保持不变。
- iter 4 (score=86.940, structure=forward_progress + height_penalty + lateral_drift_penalty + upright_stability): 4. selected_level：Level 1——高度惩罚职责合理但系数偏小（episode_sum_mean 仅 -37.7，active_rate 5.5%），仅需加大惩罚幅度即可预期改善提前终止。 | 5. selected_intervention：只修改 height_penalty 组件，将边界外惩罚系数从 20.0 提升至 60.0。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
