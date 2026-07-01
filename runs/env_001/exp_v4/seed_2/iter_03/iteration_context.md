# Iteration Context

## Recommended Action
**revert** — Best score (-108.58) is higher than current (-112.24). The main change was increasing progress_reward coefficient from 10 to 50 and adding distance_anchor. The increase likely caused oscillation (goal_near_oscillation) and the anchor added negative bias. Revert to best_reward.py coefficients (progress=10, no distance_anchor) and only tune stability_penalty or landing_shaping slightly.

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_bonus + stability_penalty | -108.58 | -108.58 | 0.00 | 72.00 | progress_reward=0.160 soft_landing_bonus=0.012 stability_penalty=-0.342 | new_best |
| 2 | distance_anchor + landing_shaping + progress_reward + stability_penalty | -112.24 | -108.58 | -3.66 | 72.00 | distance_anchor=-0.485 landing_shaping=0.012 progress_reward=0.802 stability_penalty=-0.406 | no_meaningful_improvement |

## Expert Cards
## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

## high_reward_without_success
- signal: generated_reward improves but external score stays poor
- risk: policy optimizes the custom reward but not the real task
- fix: reduce exploitable terms; add constraints tied to actual task progress and stable outcome

## Stable Lessons (from previous iterations)
- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.485057 | 0.485057 | 1.000000 | -0.861736 | -0.000282 |
| landing_shaping | 0.011906 | 0.011906 | 0.007295 | 0.000000 | 2.817319 |
| progress_reward | 0.802278 | 0.848786 | 0.999995 | -2.037974 | 2.122748 |
| stability_penalty | -0.406204 | 0.406204 | 1.000000 | -3.496580 | -0.000000 |
| total_reward | -0.077077 | 0.457257 | 1.000000 | -5.329068 | 2.802108 |
| generated_reward | -0.077077 | 0.457257 | 1.000000 | -5.329068 | 2.802108 |
| original_env_reward | -1.598657 | 2.439755 | 1.000000 | -100.000000 | 139.652583 |
