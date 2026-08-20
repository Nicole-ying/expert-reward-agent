# Fresh Restart Evidence

- target_score: 3000.000
- best_score_so_far: 5.500

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| descent_early_penalty + forward_progress + lateral_drift_hinge + roll_pitch_hinge | 1 | 5.500 | 5.500 | unsolved |
| forward_progress + lateral_drift_hinge + roll_pitch_hinge | 2 | -75.450 | -75.450 | unsolved |
| forward_progress + height_boundary_hinge + lateral_drift_hinge + roll_pitch_hinge | 1 | -186.370 | -186.370 | unsolved |
| action_cost + angular_factor + forward_reward + gate + height_factor + lateral_factor | 1 | -345.870 | -345.870 | unsolved |
| forward_reward + height_high_penalty + height_low_penalty + lateral_penalty + pitch_penalty + roll_penalty | 2 | -415.210 | -845.860 | unsolved |
| forward_progress + survival_bonus | 1 | -585.810 | -585.810 | unsolved |

## Previous interventions

- iter 2 (score=-75.450, structure=forward_progress + lateral_drift_hinge + roll_pitch_hinge): 4. `selected_level`：Level 2 结构变换，因为需要改变 upright_gate 的核心数学形态（收紧阈值）来消除误报，而非简单尺度调整。 | 5. `selected_intervention`：仅修改 upright_gate，将全开阈值从 0.5 提高到 0.8，零门阈值从 0.2 提高到 0.4，使 forward_progress 只在接近直立（body_up_z > 0.4）时才获得奖励，并且在 body_up_z 在 0.4~0.8 之间线性插值。
- iter 3 (score=-186.370, structure=forward_progress + height_boundary_hinge + lateral_drift_hinge + roll_pitch_hinge): 4. `selected_level`: Level 2 — signal gap for height penalty confirmed by termination rate and gate suppression; add one new component. | 5. `selected_intervention`: Introduce `height_boundary_hinge`: a hinge penalty on body_z outside [0.3, 0.9] with weight 0.8, providing direct negative feedback for height excursions.
- iter 4 (score=5.500, structure=descent_early_penalty + forward_progress + lateral_drift_hinge + roll_pitch_hinge): 4. `selected_level`：Level 2 structure transformation — replace height_boundary_hinge with descent_early_penalty to supply missing predictive signal and remove conflicting standalone penalty. | 5. `selected_intervention`：Remove height_boundary_hinge entirely; add descent_early_penalty that penalizes downward velocity when body_z is below a safe margin (0.5), using obs[15].
- iter 5 (score=-345.870, structure=action_cost + angular_factor + forward_reward + gate + height_factor + lateral_factor): 4. selected_level: Level 3—same skeleton family (forward + independent penalties) has persisted 4 iterations and best score remains <0.2% of target, fulfilling the REBUILD condition. | 5. selected_intervention: Replace all independent constraints with a geometric‑mean health gate (height, upright, lateral, angular including yaw) multiplied by a convex forward‑speed reward (velocity²), plus a tiny actio
- iter 7 (score=-415.210, structure=forward_reward + height_high_penalty + height_low_penalty + lateral_penalty + pitch_penalty + roll_penalty): 4. selected_level: Level 3 REBUILD，基于历史最优 iter 4 方向（无门控前进 + 轻量运动惩罚），移除生存奖励，引入高度边界惩罚，采用 dense_state_signal 正激励+ hinge 惩罚的主信号骨架。 | 5. selected_intervention: 重新构建奖励为 forward_reward(系数 2.0)、lateral_penalty(hinge,容忍 0.2)、roll/pitch_penalty(hinge,容忍 0.5)、height_low/height_high_penalty(hinge,阈 0.3/0.9)，完全移除 survival_bonus 及任何门控。
- iter 8 (score=-845.860, structure=forward_reward + height_high_penalty + height_low_penalty + lateral_penalty + pitch_penalty + roll_penalty): 4. `selected_level`：Level 2（结构变换）——累积记录显示连续 3 轮未刷新 best，当前“纯线性前进+多项独立惩罚”骨架持续失效；需要在前进奖励中引入 uprightness gate 作为局部反馈，将二值安全条件变为连续的 bounded factor，符合“sparse→dense”模式。 | 5. `selected_intervention`：将 `forward_reward` 组件从 `2.0 * forward_vel` 修改为 `2.0 * forward_vel * uprightness_gate`，其中 `uprightness_gate` 由四元数计算并通过 hinge 函数映射为 [0,1] 连续值；其余惩罚组件保持不变，仅此一个组件变更。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
