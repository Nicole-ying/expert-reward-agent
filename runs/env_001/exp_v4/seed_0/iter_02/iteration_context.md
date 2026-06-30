# Iteration Context

## Recommended Action
**tune** — 当前骨架仅运行1轮，得分-111.31远低于目标200。best_reward与previous_reward代码完全相同，因此无revert必要。主要问题：stability_penalty系数过高（angle_penalty=0.5, angular_vel_penalty=0.3, speed_penalty=0.2）导致惩罚主导，progress_reward系数10.0相对不足。建议降低stability_penalty系数（如angle_penalty=0.2, angular_vel_penalty=0.1, speed_penalty=0.1）并提高progress_reward系数（如20.0），以平衡驱动与稳定。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_reward + soft_landing_bonus + stability_penalty | -111.31 | -111.31 | 0.00 | 74.10 | energy_penalty=-0.008 progress_reward=0.161 soft_landing_bonus=0.010 stability_penalty=-0.241 | new_best |

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
| energy_penalty | -0.007926 | 0.007926 | 0.158522 | -0.050000 | -0.000000 |
| progress_reward | 0.161320 | 0.170623 | 0.999990 | -0.416742 | 0.421166 |
| soft_landing_bonus | 0.010106 | 0.010106 | 0.005053 | 0.000000 | 2.000000 |
| stability_penalty | -0.241067 | 0.241067 | 1.000000 | -3.368588 | -0.000001 |
| total_reward | -0.077567 | 0.102093 | 1.000000 | -3.579126 | 2.000559 |
| generated_reward | -0.077567 | 0.102093 | 1.000000 | -3.579126 | 2.000559 |
| original_env_reward | -1.576537 | 2.296065 | 1.000000 | -100.000000 | 133.854119 |
