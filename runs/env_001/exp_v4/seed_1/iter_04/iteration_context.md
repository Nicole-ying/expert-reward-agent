# Iteration Context

## Recommended Action
**revert** — Best score (-111.64) is higher than current (-115.99). The only change from best was reducing angle_penalty weight from 0.3 to 0.25 and increasing soft_landing_proxy from 2.0 to 2.5. The reduction in stability penalty likely increased oscillation (goal_near_oscillation) without improving landing success. Revert to best coefficients (angle_penalty=0.3, soft_landing_proxy=2.0) and consider increasing progress_delta_reward coefficient to better counteract stability penalty.

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -111.64 | -111.64 | 0.00 | 73.60 | energy_penalty=-0.011 progress_delta_reward=0.161 soft_landing_proxy=0.008 stability_penalty=-0.549 | new_best |
| 2 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -120.28 | -111.64 | -8.64 | 73.40 | energy_penalty=-0.009 progress_delta_reward=0.324 soft_landing_proxy=0.015 stability_penalty=-0.222 | no_meaningful_improvement |
| 3 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -115.99 | -111.64 | -4.35 | 73.60 | energy_penalty=-0.010 progress_delta_reward=0.161 soft_landing_proxy=0.010 stability_penalty=-0.549 | no_meaningful_improvement |

## Expert Cards
## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

## Stable Lessons (from previous iterations)
- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.
- soft_landing_proxy with strict conditions may be too sparse to provide useful signal
- Increasing progress_delta_reward coefficient without adjusting stability can cause oscillation and score drop.
- Relaxing soft_landing_proxy conditions did not increase its trigger rate; it remains too sparse.

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| energy_penalty | -0.010344 | 0.010344 | 0.103443 | -0.100000 | 0.000000 |
| progress_delta_reward | 0.161119 | 0.170508 | 0.999998 | -0.396840 | 0.424094 |
| soft_landing_proxy | 0.010212 | 0.010212 | 0.004085 | 0.000000 | 2.500000 |
| stability_penalty | -0.548776 | 0.548776 | 1.000000 | -1.986341 | -0.000001 |
| total_reward | -0.387790 | 0.407241 | 1.000000 | -2.225371 | 2.500824 |
| generated_reward | -0.387790 | 0.407241 | 1.000000 | -2.225371 | 2.500824 |
| original_env_reward | -1.645185 | 2.345848 | 1.000000 | -100.000000 | 126.101728 |
