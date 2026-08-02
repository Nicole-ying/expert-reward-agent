# Fresh Restart Evidence

- target_score: 300.000
- best_score_so_far: -52.190

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| energy_penalty + forward_reward + hinge_penalty | 2 | -52.190 | -52.190 | unsolved |
| energy_penalty + forward_reward | 1 | -59.500 | -59.500 | unsolved |
| balance_penalty + forward_progress | 1 | -59.970 | -59.970 | unsolved |
| balance_penalty + forward_reward + terrain_gate + terrain_roughness | 1 | -74.850 | -74.850 | unsolved |
| air_stability_penalty + balance_penalty + forward_progress | 1 | -86.300 | -86.300 | unsolved |
| air_stability_penalty + balance_penalty + forward_reward + terrain_gate + terrain_roughness | 1 | -95.840 | -95.840 | unsolved |

## Previous interventions

- iter 7 (score=-52.190, structure=energy_penalty + forward_reward + hinge_penalty): 因此，本轮修改选择 **Level 2 结构变换 — 添加一个组件**：重新引入基于 hull_angle 的 hinge_penalty。该组件在 |hull_angle| 超过安全阈值 0.3 rad 后施加线性惩罚，直接告诫 agent 保持低倾斜角。配合已有的双因子门控（角度 + 角速度继续压低前进奖励），形成“前进减速 + 直接姿态惩罚”的双重防护，更清晰地表达“避免摔倒”的目标。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
