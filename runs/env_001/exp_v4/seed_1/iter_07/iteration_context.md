# Iteration Context

## Recommended Action
**tune** — 当前骨架已运行4轮，得分稳定在-111左右，远低于目标200。best_reward与previous_reward代码完全相同，因此无需revert。主要问题是stability_penalty权重过大（-0.5 speed, -0.3 angle, -0.1 angular_vel），导致agent过于保守，progress_delta_reward系数10.0不足以驱动有效学习。soft_landing_proxy触发率极低（0.4%），几乎无贡献。建议降低stability_penalty权重（如speed系数降至-0.2），并适当提高progress_delta_reward系数（如20.0），同时考虑放宽soft_landing_proxy条件或替换为更密集的着陆奖励。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -111.64 | -111.64 | 0.00 | 73.60 | energy_penalty=-0.011 progress_delta_reward=0.161 soft_landing_proxy=0.008 stability_penalty=-0.549 | new_best |
| 2 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -120.28 | -111.64 | -8.64 | 73.40 | energy_penalty=-0.009 progress_delta_reward=0.324 soft_landing_proxy=0.015 stability_penalty=-0.222 | no_meaningful_improvement |
| 3 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -115.99 | -111.64 | -4.35 | 73.60 | energy_penalty=-0.010 progress_delta_reward=0.161 soft_landing_proxy=0.010 stability_penalty=-0.549 | no_meaningful_improvement |
| 4 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -114.12 | -111.64 | -2.48 | 73.60 | energy_penalty=-0.010 progress_delta_reward=0.193 soft_landing_proxy=0.008 stability_penalty=-0.550 | unsolved_stagnation_fresh_restart |
| 5 | action_penalty + progress_reward + soft_landing_bonus + stability_penalty | -110.80 | -111.64 | 0.84 | 73.60 | action_penalty=-0.006 progress_reward=0.081 soft_landing_bonus=0.010 stability_penalty=-0.344 | no_meaningful_improvement |
| 6 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -111.64 | -111.64 | 0.00 | 73.60 | energy_penalty=-0.011 progress_delta_reward=0.161 soft_landing_proxy=0.008 stability_penalty=-0.549 | no_meaningful_improvement |

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
- soft_landing_proxy with strict conditions may be too sparse to provide useful signal
- Increasing progress_delta_reward coefficient without adjusting stability can cause oscillation and score drop.
- Relaxing soft_landing_proxy conditions did not increase its trigger rate; it remains too sparse.
- Reducing stability penalty weight can lead to oscillation and score regression.
- Increasing soft_landing_proxy reward does not improve trigger rate if conditions remain strict.
- stability_penalty weights should prioritize speed over angle to avoid dominance

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
