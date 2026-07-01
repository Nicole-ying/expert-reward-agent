# Iteration Context

## Recommended Action
**revert** — Current score (-110.36) is worse than best (-111.64) but best_reward.py (score -111.64) uses progress_delta_reward with coefficient 10.0 and energy_penalty -0.1, while current uses progress_reward with coefficient 5.0 and action_penalty -0.05. The best configuration achieved higher score despite similar skeleton. Revert to best_reward.py coefficients and components, then consider tuning stability_penalty weights to reduce dominance.

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -111.64 | -111.64 | 0.00 | 73.60 | energy_penalty=-0.011 progress_delta_reward=0.161 soft_landing_proxy=0.008 stability_penalty=-0.549 | new_best |
| 2 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -120.28 | -111.64 | -8.64 | 73.40 | energy_penalty=-0.009 progress_delta_reward=0.324 soft_landing_proxy=0.015 stability_penalty=-0.222 | no_meaningful_improvement |
| 3 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -115.99 | -111.64 | -4.35 | 73.60 | energy_penalty=-0.010 progress_delta_reward=0.161 soft_landing_proxy=0.010 stability_penalty=-0.549 | no_meaningful_improvement |
| 4 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -114.12 | -111.64 | -2.48 | 73.60 | energy_penalty=-0.010 progress_delta_reward=0.193 soft_landing_proxy=0.008 stability_penalty=-0.550 | unsolved_stagnation_fresh_restart |
| 5 | action_penalty + progress_reward + soft_landing_bonus + stability_penalty | -110.80 | -111.64 | 0.84 | 73.60 | action_penalty=-0.006 progress_reward=0.081 soft_landing_bonus=0.010 stability_penalty=-0.344 | no_meaningful_improvement |
| 6 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -111.64 | -111.64 | 0.00 | 73.60 | energy_penalty=-0.011 progress_delta_reward=0.161 soft_landing_proxy=0.008 stability_penalty=-0.549 | no_meaningful_improvement |
| 7 | energy_penalty + landing_shaping + progress_delta_reward + stability_penalty | -108.86 | -111.64 | 2.79 | 73.60 | energy_penalty=-0.010 landing_shaping=0.085 progress_delta_reward=0.324 stability_penalty=-0.219 | unsolved_stagnation_fresh_restart |
| 8 | action_penalty + progress_reward + soft_landing_bonus + stability_penalty | -110.36 | -111.64 | 1.28 | 73.60 | action_penalty=-0.007 progress_reward=0.081 soft_landing_bonus=0.010 stability_penalty=-0.243 | no_meaningful_improvement |

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
- Reducing stability penalty weight can lead to oscillation and score regression.
- Increasing soft_landing_proxy reward does not improve trigger rate if conditions remain strict.
- stability_penalty weights should prioritize speed over angle to avoid dominance
- stability_penalty coefficients must be balanced to avoid dominating progress signal
- soft_landing_proxy with strict conditions is too sparse; consider relaxing conditions or using a continuous proxy

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| action_penalty | -0.007419 | 0.007419 | 0.148378 | -0.050000 | 0.000000 |
| progress_reward | 0.080831 | 0.085478 | 0.999990 | -0.203761 | 0.211999 |
| soft_landing_bonus | 0.009917 | 0.009917 | 0.004959 | 0.000000 | 2.000000 |
| stability_penalty | -0.242780 | 0.242780 | 1.000000 | -3.219305 | -0.000001 |
| total_reward | -0.159451 | 0.178346 | 1.000000 | -3.362454 | 2.000603 |
| generated_reward | -0.159451 | 0.178346 | 1.000000 | -3.362454 | 2.000603 |
| original_env_reward | -1.577038 | 2.307488 | 1.000000 | -100.000000 | 139.173759 |
