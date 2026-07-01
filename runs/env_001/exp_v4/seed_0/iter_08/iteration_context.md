# Iteration Context

## Recommended Action
**revert** — 当前得分-118.81远低于best得分-18.56（差距>100）。对比代码：best使用progress_scale=80.0，当前仅10.0；best的stability_penalty权重总和约0.035，当前约0.5（speed_scale=0.3, angle_scale=0.2, angular_vel_scale=0.1乘以stability_scale=0.5），导致stability_penalty主导；best的soft_landing_proxy使用连续塑形（多级奖励），当前仅单一阈值2.0且触发率极低。建议恢复到best配置，仅做小幅调整。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -119.87 | -119.87 | 0.00 | 73.90 | action_penalty=-0.002 angle_penalty=-0.002 angular_vel_penalty=-0.001 progress_delta_reward=0.054 soft_landing_proxy=0.004 | new_best |
| 2 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -27.09 | -27.09 | 0.00 | 1000.00 | action_penalty=-0.004 angle_penalty=-0.002 angular_vel_penalty=-0.001 progress_delta_reward=0.031 soft_landing_proxy=0.957 | new_best |
| 3 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -18.56 | -18.56 | 0.00 | 800.60 | action_penalty=-0.004 angle_penalty=-0.004 angular_vel_penalty=-0.002 progress_delta_reward=1.416 soft_landing_proxy=0.511 | new_best |
| 4 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -48.55 | -18.56 | -29.99 | 1000.00 | action_penalty=-0.004 angle_penalty=-0.003 angular_vel_penalty=-0.001 progress_delta_reward=0.915 soft_landing_proxy=1.793 | no_meaningful_improvement |
| 5 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -18.56 | -18.56 | 0.00 | 800.60 | action_penalty=-0.004 angle_penalty=-0.004 angular_vel_penalty=-0.002 progress_delta_reward=1.416 soft_landing_proxy=0.511 | no_meaningful_improvement |
| 6 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -48.55 | -18.56 | -29.99 | 1000.00 | action_penalty=-0.004 angle_penalty=-0.003 angular_vel_penalty=-0.001 progress_delta_reward=0.915 soft_landing_proxy=1.793 | unsolved_stagnation_fresh_restart |
| 7 | energy_penalty + progress_reward + soft_landing_bonus + stability_penalty | -118.81 | -18.56 | -100.26 | 74.00 | energy_penalty=-0.008 progress_reward=0.162 soft_landing_bonus=0.008 stability_penalty=-0.167 | no_meaningful_improvement |

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
- stability_penalty权重过大（-0.1086）会主导总奖励，抑制progress信号
- soft_landing_proxy触发率过低（0.0043）需放宽条件或增加权重
- soft_landing_proxy thresholds should be tightened to avoid contact_reward_hacking
- soft_landing_proxy触发率低（0.21），需放宽阈值或增加权重以提升学习信号
- progress_delta_reward系数80.0有效驱动学习，可保持
- soft_landing_proxy thresholds must be tight (<=0.3) to avoid contact_reward_hacking
- soft_landing_proxy weight should not exceed 3.0 when thresholds are tight

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| energy_penalty | -0.008259 | 0.008259 | 0.082590 | -0.100000 | 0.000000 |
| progress_reward | 0.161521 | 0.170844 | 0.999994 | -0.418984 | 0.421667 |
| soft_landing_bonus | 0.007932 | 0.007932 | 0.003966 | 0.000000 | 2.000000 |
| stability_penalty | -0.166909 | 0.166909 | 1.000000 | -0.776104 | -0.000000 |
| total_reward | -0.005714 | 0.078491 | 1.000000 | -1.064015 | 2.005625 |
| generated_reward | -0.005714 | 0.078491 | 1.000000 | -1.064015 | 2.005625 |
| original_env_reward | -1.640570 | 2.317459 | 1.000000 | -100.000000 | 128.534317 |
