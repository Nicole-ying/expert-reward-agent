# Iteration Context

## Recommended Action
**tune** — 当前骨架仅运行1轮，得分-111.98远低于目标200。best_reward与previous_reward完全相同，因此无需revert。主要问题是progress_reward系数过低（2.0），导致学习信号弱，而stability_penalty系数相对过大，主导了总奖励。建议增大progress_scale（如10.0），同时降低stability_penalty各系数（如angle_penalty=0.2, angular_vel_penalty=0.1, speed_penalty=0.1），并考虑提高landing_bonus或使其更易触发。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -111.98 | -111.98 | 0.00 | 70.60 | energy_penalty=-0.012 landing_bonus=0.005 progress_reward=0.032 stability_penalty=-0.243 | new_best |

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
| energy_penalty | -0.011773 | 0.011773 | 0.117730 | -0.100000 | -0.000000 |
| landing_bonus | 0.005296 | 0.005296 | 0.005296 | 0.000000 | 1.000000 |
| progress_reward | 0.032315 | 0.034176 | 0.999995 | -0.082247 | 0.084779 |
| stability_penalty | -0.243358 | 0.243358 | 1.000000 | -3.432804 | -0.000000 |
| total_reward | -0.217520 | 0.227378 | 1.000000 | -3.572376 | 1.000406 |
| generated_reward | -0.217520 | 0.227378 | 1.000000 | -3.572376 | 1.000406 |
| original_env_reward | -1.587230 | 2.302484 | 1.000000 | -100.000000 | 128.588050 |
