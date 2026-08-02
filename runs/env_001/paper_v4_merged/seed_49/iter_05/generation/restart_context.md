# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -113.710

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| angle_penalty + angvel_penalty + efficiency + progress + soft_landing | 2 | -113.710 | -115.680 | unsolved |
| angle_penalty + efficiency + failure_penalty + progress + success_bonus | 1 | -120.200 | -120.200 | unsolved |
| angle_penalty + angvel_penalty + efficiency + failure_penalty + progress + soft_landing | 1 | -222.060 | -222.060 | unsolved |

## Previous interventions

- iter 2 (score=-115.680, structure=angle_penalty + angvel_penalty + efficiency + progress + soft_landing): 4. **selected_level**：Level 2 — 结构变换，触发基础是“progress的数学形态为unbounded线性正奖励，且外部得分在shaped奖励持续为正的情况下仍为负”，属于“proxy 提高但外部分数不升”的证据模式，需要对主正向信号施加边界约束。 | 5. **selected_intervention**：唯一目标组件是`progress`。修改方式：在计算`delta_dist`后，乘入一个基于垂直速度的安全下降门控因子`gate`。当下降速度（-vy）超过`max_safe_vy=0.5`时，`gate`线性衰减至0，从而削弱高速下降时的progress奖励强度，其余组件保持不变。
- iter 3 (score=-222.060, structure=angle_penalty + angvel_penalty + efficiency + failure_penalty + progress + soft_landing): 4. `selected_level`：Level 2，因观测到信号缺口（灾难性失败无覆盖），且上一轮尺度调整并未改变行为，符合“缺职责 → add 新组件”的结构变换条件。 | 5. `selected_intervention`：新增 `terminal_failure_penalty` 组件，当观测到失败状态（水平越界或垂直高度过低）时给予较大负惩罚，其余组件保持不变。
- iter 4 (score=-120.200, structure=angle_penalty + efficiency + failure_penalty + progress + success_bonus): 4. **selected_level**：Level 3 rebuild — the same skeleton family failed for 3 consecutive rounds and the best score never exceeded -113, with pre‑judgement all ❌. | 5. **selected_intervention**：design a new skeleton based on improvement_delta (distance reduction + encouraged descent) as the main progress signal, add separate success_bonus (soft‑landing condition) and failure_penalty

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
