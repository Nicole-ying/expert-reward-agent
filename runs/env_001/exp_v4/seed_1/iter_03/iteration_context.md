# Iteration Context

## Recommended Action
**revert** — Best score (-111.64) is higher than current (-120.28) by 8.64. The only changes from best were: progress_delta_reward coefficient 10->20, stability_penalty weights reduced (speed 0.5->0.2, angle 0.3->0.1, angular_vel 0.1->0.05), soft_landing_proxy conditions relaxed (dist 0.3->0.5, speed 0.2->0.3, angle 0.2->0.3) and reward 2.0->3.0. These changes caused regression. Revert to best coefficients and only make small adjustments.

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -111.64 | -111.64 | 0.00 | 73.60 | energy_penalty=-0.011 progress_delta_reward=0.161 soft_landing_proxy=0.008 stability_penalty=-0.549 | new_best |
| 2 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -120.28 | -111.64 | -8.64 | 73.40 | energy_penalty=-0.009 progress_delta_reward=0.324 soft_landing_proxy=0.015 stability_penalty=-0.222 | no_meaningful_improvement |

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

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| energy_penalty | -0.008773 | 0.008773 | 0.087728 | -0.100000 | 0.000000 |
| progress_delta_reward | 0.323544 | 0.342067 | 0.999996 | -0.805220 | 0.841752 |
| soft_landing_proxy | 0.014702 | 0.014702 | 0.004901 | 0.000000 | 3.000000 |
| stability_penalty | -0.221762 | 0.221762 | 1.000000 | -0.887166 | -0.000000 |
| total_reward | 0.107712 | 0.192744 | 1.000000 | -1.426826 | 3.034254 |
| generated_reward | 0.107712 | 0.192744 | 1.000000 | -1.426826 | 3.034254 |
| original_env_reward | -1.662235 | 2.341743 | 1.000000 | -100.000000 | 138.437293 |
