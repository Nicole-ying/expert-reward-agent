# Iteration Context

## Recommended Action
**tune** — 当前骨架仅迭代2轮，得分从-102提升至-26.5，趋势上升。best_reward与previous_reward代码完全相同，因此无需revert。但得分仍远低于目标200，且progress_delta_reward信号弱（mean 0.009），stability_penalty均值-0.049可能压制了移动。建议：增大progress_delta_reward系数（如从5.0增至10.0），同时降低stability_penalty中的角度惩罚系数（如从-0.3降至-0.1），以鼓励更多探索。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + stability_penalty | -102.17 | -102.17 | 0.00 | 33.00 | energy_penalty=-0.016 progress_delta_reward=-0.005 stability_penalty=-0.134 | new_best |
| 2 | energy_penalty + progress_delta_reward + stability_penalty | -26.51 | -26.51 | 0.00 | 1600.00 | energy_penalty=-0.010 progress_delta_reward=0.009 stability_penalty=-0.049 | new_best |

## Expert Cards
## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

## generated_reward_negative_average
- signal: total generated reward mean is negative during most training steps
- risk: agent receives mostly punitive feedback and may not discover useful behavior
- fix: rebalance progress and penalty terms; avoid large always-on penalties in early versions

## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## Stable Lessons (from previous iterations)
- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| energy_penalty | -0.009890 | 0.009890 | 1.000000 | -0.040000 | -0.000002 |
| progress_delta_reward | 0.009211 | 0.064857 | 1.000000 | -1.323389 | 1.193558 |
| stability_penalty | -0.048647 | 0.048647 | 1.000000 | -1.096758 | -0.000068 |
| total_reward | -0.049326 | 0.082674 | 1.000000 | -1.729570 | 1.015530 |
| generated_reward | -0.049326 | 0.082674 | 1.000000 | -1.729570 | 1.015530 |
| original_env_reward | -0.919997 | 0.990828 | 1.000000 | -100.000000 | 0.597136 |
