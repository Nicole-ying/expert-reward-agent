# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: 108.450

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| approach_landing_reward + distance_cost + stability_cost | 4 | 108.450 | 108.450 | unsolved |
| distance_cost + soft_landing_bonus + stability_cost | 1 | -119.560 | -119.560 | unsolved |
| approach_landing_reward + distance_cost + engine_penalty + stability_cost | 1 | -125.740 | -125.740 | unsolved |

## Previous interventions

- iter 2 (score=-98.890, structure=approach_landing_reward + distance_cost + stability_cost): 4. `selected_level`：Level 2 — `sparse_to_dense`：完成 bonus 的结构性稀疏（0.3% << 1%）明确要求将硬性二值事件替换为连续过程证据，而非仅调系数。 | 5. `selected_intervention`：将 `soft_landing_bonus` 从硬性六条件联合判定，转换为由接近度、速度、姿态角和支撑腿接触的连续有界指数得分之和构成的 `approach_landing_reward`。
- iter 3 (score=-87.910, structure=approach_landing_reward + distance_cost + stability_cost): `selected_level`：Level 2 — `dense_to_task_event`。稠密代理已教会agent到达着陆垫（ep长度从68→752，不再crash），但现在主导奖励并阻止高效完成。证据模式完全匹配："dense proxy forming medium score plateau"和"proxy提高但外部任务不升"。根据知识库指引：应reduce/difference/localize该dense proxy。 | `selected_intervention`：将approach_landing_reward从**持续性状态值**（state-value：`e^(-dist²)`、`e^(-vel²)`、contact连续值）重构为**改善量**（improvement-based：`max(0, prev - curr)`形式），并将velocity improvement以proximity gate约束（仅近端减速获奖励）。四个子项统一改为
- iter 5 (score=108.450, structure=approach_landing_reward + distance_cost + stability_cost): selected_level**: Level 1 — distance_cost role and monotonic math form are reasonable, but its coefficient makes penalty dominate progress at 1.94≫0.5, a clear scale problem per penalty_magnitude_control evidence. | selected_intervention**: Reduce w_dist from 1.0 to 0.08 (12.5× reduction, exceeding the 10× minimum for penalty dominance), applied to best code (iter 3 base with proximity_improvement=5.0), no other component changed.
- iter 6 (score=-125.740, structure=approach_landing_reward + distance_cost + engine_penalty + stability_cost): `selected_level`：Level 2。触发条件：improvement型proxy（proxy_to_completion_alignment模式）被策略系统性套利——外部任务完成（成功着陆）但效率极差（episode过长），且缺失引擎使用约束这一任务明确要求。 | `selected_intervention`：新增`engine_penalty`组件——对任何非零动作（action 1/2/3使用引擎）施加固定惩罚，惩罚"使用引擎推力"这一行为本身。这是当前奖励完全缺失的任务维度。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
