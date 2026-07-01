# Iteration Context

## Recommended Action
**tune** — 当前骨架仅运行1轮，得分-111.64远低于目标200。best_reward与previous_reward完全相同，无需revert。stability_penalty均值-0.549，绝对值大于progress_delta_reward均值0.161，导致总奖励为负，agent可能过于保守。建议降低stability_penalty系数（如speed系数从0.5降至0.2），同时提高progress_delta_reward系数（如从10.0增至20.0）以增强接近目标的驱动力。soft_landing_proxy触发率极低，可放宽条件（如near_target阈值从0.3增至0.5）或暂时移除。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -111.64 | -111.64 | 0.00 | 73.60 | energy_penalty=-0.011 progress_delta_reward=0.161 soft_landing_proxy=0.008 stability_penalty=-0.549 | new_best |

## Expert Cards
## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

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
| energy_penalty | -0.010656 | 0.010656 | 0.106562 | -0.100000 | 0.000000 |
| progress_delta_reward | 0.161298 | 0.170642 | 0.999994 | -0.413573 | 0.424454 |
| soft_landing_proxy | 0.008243 | 0.008243 | 0.004121 | 0.000000 | 2.000000 |
| stability_penalty | -0.549313 | 0.549313 | 1.000000 | -2.490366 | -0.000000 |
| total_reward | -0.390428 | 0.405961 | 1.000000 | -2.757616 | 2.001656 |
| generated_reward | -0.390428 | 0.405961 | 1.000000 | -2.757616 | 2.001656 |
| original_env_reward | -1.632736 | 2.345054 | 1.000000 | -100.000000 | 134.090720 |
