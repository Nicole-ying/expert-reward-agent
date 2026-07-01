# Iteration Context

## Recommended Action
**tune** — 当前得分-27.09，较上一轮-119.87大幅提升，但距离目标200仍远。best_reward与previous_reward代码完全相同，无需revert。soft_landing_proxy触发率60.4%但mean 0.957，可能条件过宽导致contact_reward_hacking。progress_delta_reward信号偏弱（mean 0.03），建议增大progress_scale至50-100。stability_penalty权重已较低，暂不调整。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -119.87 | -119.87 | 0.00 | 73.90 | action_penalty=-0.002 angle_penalty=-0.002 angular_vel_penalty=-0.001 progress_delta_reward=0.054 soft_landing_proxy=0.004 | new_best |
| 2 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -27.09 | -27.09 | 0.00 | 1000.00 | action_penalty=-0.004 angle_penalty=-0.002 angular_vel_penalty=-0.001 progress_delta_reward=0.031 soft_landing_proxy=0.957 | new_best |

## Expert Cards
## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

## high_reward_without_success
- signal: generated_reward improves but external score stays poor
- risk: policy optimizes the custom reward but not the real task
- fix: reduce exploitable terms; add constraints tied to actual task progress and stable outcome

## contact_reward_hacking
- signal: contact-related reward triggers without good external score
- risk: agent exploits contact flags without safe task completion
- fix: require near target, low speed, stable angle, and both supports before using contact

## Stable Lessons (from previous iterations)
- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.
- stability_penalty权重过大（-0.1086）会主导总奖励，抑制progress信号
- soft_landing_proxy触发率过低（0.0043）需放宽条件或增加权重

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| action_penalty | -0.003832 | 0.003832 | 0.766413 | -0.005000 | 0.000000 |
| angle_penalty | -0.001595 | 0.001595 | 1.000000 | -0.044168 | -0.000000 |
| angular_vel_penalty | -0.000608 | 0.000608 | 0.999957 | -0.038402 | -0.000000 |
| progress_delta_reward | 0.030585 | 0.033920 | 0.998839 | -0.215056 | 0.284379 |
| soft_landing_proxy | 0.957299 | 0.957299 | 0.603849 | 0.000000 | 2.000000 |
| speed_penalty | -0.005545 | 0.005545 | 0.999976 | -0.048127 | -0.000000 |
| stability_penalty | -0.007748 | 0.007748 | 1.000000 | -0.095339 | -0.000000 |
| total_reward | 0.976304 | 0.981377 | 1.000000 | -0.282885 | 2.004612 |
| generated_reward | 0.976304 | 0.981377 | 1.000000 | -0.282885 | 2.004612 |
| original_env_reward | -0.259551 | 1.842360 | 1.000000 | -100.000000 | 129.273210 |
