# Iteration Context

## Recommended Action
**revert** — 当前得分-48.55远低于best得分-18.56。对比代码，current将soft_landing_proxy阈值从0.3/0.3/0.2放宽到0.5/0.5/0.3，权重从3.0提升到5.0，导致contact_reward_hacking（触发率0.45但得分下降）。best的收紧阈值和较低权重更有效。建议revert到best配置，仅微调。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -119.87 | -119.87 | 0.00 | 73.90 | action_penalty=-0.002 angle_penalty=-0.002 angular_vel_penalty=-0.001 progress_delta_reward=0.054 soft_landing_proxy=0.004 | new_best |
| 2 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -27.09 | -27.09 | 0.00 | 1000.00 | action_penalty=-0.004 angle_penalty=-0.002 angular_vel_penalty=-0.001 progress_delta_reward=0.031 soft_landing_proxy=0.957 | new_best |
| 3 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -18.56 | -18.56 | 0.00 | 800.60 | action_penalty=-0.004 angle_penalty=-0.004 angular_vel_penalty=-0.002 progress_delta_reward=1.416 soft_landing_proxy=0.511 | new_best |
| 4 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -48.55 | -18.56 | -29.99 | 1000.00 | action_penalty=-0.004 angle_penalty=-0.003 angular_vel_penalty=-0.001 progress_delta_reward=0.915 soft_landing_proxy=1.793 | no_meaningful_improvement |

## Expert Cards
## contact_reward_hacking
- signal: contact-related reward triggers without good external score
- risk: agent exploits contact flags without safe task completion
- fix: require near target, low speed, stable angle, and both supports before using contact

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

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| action_penalty | -0.003787 | 0.003787 | 0.757431 | -0.005000 | 0.000000 |
| angle_penalty | -0.002525 | 0.002525 | 1.000000 | -0.044408 | -0.000000 |
| angular_vel_penalty | -0.000980 | 0.000980 | 0.999956 | -0.046792 | -0.000000 |
| progress_delta_reward | 0.914808 | 0.980319 | 0.999681 | -4.438019 | 4.486380 |
| soft_landing_proxy | 1.792524 | 1.792524 | 0.448543 | 0.000000 | 5.000000 |
| speed_penalty | -0.009157 | 0.009157 | 0.999976 | -0.051183 | -0.000000 |
| stability_penalty | -0.012663 | 0.012663 | 1.000000 | -0.099236 | -0.000000 |
| total_reward | 2.690882 | 2.754266 | 1.000000 | -4.536543 | 5.176293 |
| generated_reward | 2.690882 | 2.754266 | 1.000000 | -4.536543 | 5.176293 |
| original_env_reward | -0.557439 | 2.555350 | 1.000000 | -100.000000 | 144.167731 |
