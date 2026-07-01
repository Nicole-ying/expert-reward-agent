# Reward Memory

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
| 9 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -114.39 | -111.64 | -2.75 | 73.60 | energy_penalty=-0.010 progress_delta_reward=0.162 soft_landing_proxy=0.008 stability_penalty=-0.543 | no_meaningful_improvement |
| 10 | energy_penalty + landing_shaping + progress_delta_reward + stability_penalty | -119.88 | -111.64 | -8.23 | 73.40 | energy_penalty=-0.010 landing_shaping=0.010 progress_delta_reward=0.162 stability_penalty=-0.327 | unsolved_stagnation_fresh_restart |

## Stable Lessons

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
- progress_delta_reward coefficient of 10.0 outperforms 5.0 in this task
- energy_penalty (-0.1) is preferable to action_penalty (-0.05) for efficiency constraint
