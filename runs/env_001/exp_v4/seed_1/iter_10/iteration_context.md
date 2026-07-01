# Iteration Context

## Recommended Action
**tune** — 当前骨架已迭代5轮，得分稳定在-114左右，未突破-111.64。best_reward.py与previous_reward.py几乎相同，仅stability_penalty系数不同：best中angle_penalty权重0.3、angular_vel_penalty权重0.1，而current中分别为0.2和0.05。best得分-111.64略高于current的-114.39，但差距不大，且best也是该骨架下的得分。主要问题是stability_penalty过强（mean -0.543 vs progress 0.162），建议小幅降低stability_penalty系数，例如将speed权重从0.5降至0.4，angle_penalty从0.2降至0.15，angular_vel_penalty从0.05降至0.03，以平衡约束与进度信号。同时soft_landing_proxy触发率极低（0.4%），可考虑放宽条件或改用连续proxy。

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
| 9 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -114.39 | -111.64 | -2.75 | 73.60 | energy_penalty=-0.010 progress_delta_reward=0.162 soft_landing_proxy=0.008 stability_penalty=-0.543 | no_meaningful_improvement |

## Expert Cards
## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## sparse_completion_proxy
- signal: completion/landing bonus trigger_rate < 1%
- risk: final bonus provides little early learning guidance
- fix: replace hard bonus with smoother landing-quality shaping

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
- progress_delta_reward coefficient of 10.0 outperforms 5.0 in this task
- energy_penalty (-0.1) is preferable to action_penalty (-0.05) for efficiency constraint

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| energy_penalty | -0.010019 | 0.010019 | 0.100189 | -0.100000 | 0.000000 |
| progress_delta_reward | 0.162089 | 0.171303 | 0.999999 | -0.408383 | 0.423917 |
| soft_landing_proxy | 0.008175 | 0.008175 | 0.004088 | 0.000000 | 2.000000 |
| stability_penalty | -0.543434 | 0.543434 | 1.000000 | -1.715713 | -0.000000 |
| total_reward | -0.383189 | 0.398592 | 1.000000 | -1.864523 | 2.000926 |
| generated_reward | -0.383189 | 0.398592 | 1.000000 | -1.864523 | 2.000926 |
| original_env_reward | -1.647436 | 2.353955 | 1.000000 | -100.000000 | 124.427004 |
