# Iteration Context

## Recommended Action
**tune** — 当前骨架仅运行1轮，得分-119.87远低于目标200。best_reward与previous_reward完全相同，无需revert。主要问题是stability_penalty和speed_penalty权重过大（-0.1086和-0.1056），压制了progress_delta_reward的正信号（0.054）。soft_landing_proxy触发率极低（0.0043），条件过严。建议降低speed_penalty_weight和angle_penalty_weight，放宽soft_landing_proxy条件，并增加progress_scale以增强学习信号。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_penalty + angle_penalty + angular_vel_penalty + progress_delta_reward + soft_landing_proxy + speed_penalty | -119.87 | -119.87 | 0.00 | 73.90 | action_penalty=-0.002 angle_penalty=-0.002 angular_vel_penalty=-0.001 progress_delta_reward=0.054 soft_landing_proxy=0.004 | new_best |

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

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| action_penalty | -0.001973 | 0.001973 | 0.197301 | -0.010000 | 0.000000 |
| angle_penalty | -0.001591 | 0.001591 | 1.000000 | -0.208482 | -0.000000 |
| angular_vel_penalty | -0.001434 | 0.001434 | 0.999825 | -0.167778 | -0.000000 |
| progress_delta_reward | 0.053915 | 0.057292 | 0.999991 | -0.096340 | 0.104236 |
| soft_landing_proxy | 0.004321 | 0.004321 | 0.004321 | 0.000000 | 1.000000 |
| speed_penalty | -0.105574 | 0.105574 | 0.999853 | -0.236266 | -0.000000 |
| stability_penalty | -0.108599 | 0.108599 | 1.000000 | -0.520032 | -0.000000 |
| total_reward | -0.052336 | 0.061855 | 1.000000 | -0.616372 | 0.999728 |
| generated_reward | -0.052336 | 0.061855 | 1.000000 | -0.616372 | 0.999728 |
| original_env_reward | -1.569260 | 2.337791 | 1.000000 | -100.000000 | 125.034054 |
