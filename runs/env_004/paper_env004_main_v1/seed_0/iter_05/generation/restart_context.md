# Fresh Restart Evidence

- target_score: 3800.000
- best_score_so_far: 2663.090

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| forward_stability_reward + stability_penalty | 2 | 2663.090 | 2663.090 | unsolved |
| forward_stability_reward + stability_penalty + vertical_activity | 1 | 1338.040 | 1338.040 | unsolved |
| forward_stability_reward + stability_penalty + vertical_pushoff | 1 | 380.750 | 380.750 | unsolved |

## Previous interventions

- iter 3 (score=1338.040, structure=forward_stability_reward + stability_penalty + vertical_activity): `selected_level`：Level 2（必要信号缺失）。跳跃任务的垂直振荡动态完全没有被奖励函数表达，仅靠 forward_vel * upright_factor 无法区分跳跃步态与贴地滑行，需要在数学结构上增加一个垂直活动组件来弥补信号缺口。 | `selected_intervention`：新增 `vertical_activity` 组件——对垂直速度的绝对值进行奖励，并用同一 upright_factor 门控，确保只在直立姿态下鼓励跳跃的升降动态。系数设为 0.2，使其成为辅助信号而不会压倒主前进目标。
- iter 4 (score=380.750, structure=forward_stability_reward + stability_penalty + vertical_pushoff): 4. `selected_level`：Level 2 —— `abs(vertical_vel)` 的数学形态直接导致 proxy 与外部任务错位（奖励坠落破坏稳定），证据明确否定该形态，不是单纯尺度问题。 | 5. `selected_intervention`：将 `abs(vertical_vel)` 替换为 `max(0, vertical_vel)`，仅奖励上升阶段（主动推离），不再奖励下降阶段（被动坠落）；系数从 0.2 降至 0.15 以匹配新值域（信号范围减半，保守起步）。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
