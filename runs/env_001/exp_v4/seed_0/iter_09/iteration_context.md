# Iteration Context

## Recommended Action
**tune** — 当前骨架已迭代6轮，最佳得分-18.56，远低于目标200，且最近两轮得分重复-18.56和-48.55，陷入停滞。best_reward.py与previous_reward.py代码相同，无需revert。soft_landing_proxy触发率低（0.21），建议放宽阈值或增加权重以提升学习信号。progress_delta_reward系数80.0有效，可保持。

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
| 8 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -18.56 | -18.56 | 0.00 | 800.60 | action_penalty=-0.004 angle_penalty=-0.004 angular_vel_penalty=-0.002 progress_delta_reward=1.416 soft_landing_proxy=0.511 | no_meaningful_improvement |

## Expert Cards
## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

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
- soft_landing_proxy thresholds should be tightened to avoid contact_reward_hacking
- soft_landing_proxy触发率低（0.21），需放宽阈值或增加权重以提升学习信号
- progress_delta_reward系数80.0有效驱动学习，可保持
- soft_landing_proxy thresholds must be tight (<=0.3) to avoid contact_reward_hacking
- soft_landing_proxy weight should not exceed 3.0 when thresholds are tight
- soft_landing_proxy should use continuous shaping with multiple tiers to improve trigger rate

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| action_penalty | -0.003913 | 0.003913 | 0.782634 | -0.005000 | 0.000000 |
| angle_penalty | -0.004191 | 0.004191 | 1.000000 | -0.045368 | -0.000000 |
| angular_vel_penalty | -0.001571 | 0.001571 | 0.999930 | -0.042920 | -0.000000 |
| progress_delta_reward | 1.416445 | 1.504383 | 0.999693 | -3.933163 | 5.309043 |
| soft_landing_proxy | 0.510645 | 0.510645 | 0.213657 | 0.000000 | 3.000000 |
| speed_penalty | -0.013427 | 0.013427 | 0.999945 | -0.057515 | -0.000000 |
| stability_penalty | -0.019189 | 0.019189 | 1.000000 | -0.112147 | -0.000000 |
| total_reward | 1.903988 | 1.992772 | 1.000000 | -4.026131 | 5.230500 |
| generated_reward | 1.903988 | 1.992772 | 1.000000 | -4.026131 | 5.230500 |
| original_env_reward | -1.348851 | 3.491913 | 1.000000 | -100.000000 | 136.278567 |
