# Iteration Context

## Recommended Action
**revert** — Best score 67.07 achieved with progress_reward=50, stability_penalty lighter (angle=0.2, angular=0.1, speed=0.15), landing_shaping coefficient=3.0 and relaxed thresholds (dist=1.0, speed=1.0, angle=0.5). Current version increased progress to 80, tightened landing thresholds (dist=0.6, speed=0.5, angle=0.3) and reduced coefficient to 1.5, and slightly increased stability penalties. Score dropped to -113.70. The tightening of landing shaping made it too sparse (nonzero rate 0.0077), and the increased progress coefficient may cause oscillation. Revert to best coefficients and only make small adjustments.

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_bonus + stability_penalty | -108.58 | -108.58 | 0.00 | 72.00 | progress_reward=0.160 soft_landing_bonus=0.012 stability_penalty=-0.342 | new_best |
| 2 | distance_anchor + landing_shaping + progress_reward + stability_penalty | -112.24 | -108.58 | -3.66 | 72.00 | distance_anchor=-0.485 landing_shaping=0.012 progress_reward=0.802 stability_penalty=-0.406 | no_meaningful_improvement |
| 3 | landing_shaping + progress_reward + stability_penalty | -101.17 | -101.17 | 0.00 | 72.00 | landing_shaping=0.013 progress_reward=0.160 stability_penalty=-0.338 | new_best |
| 4 | landing_shaping + progress_reward + stability_penalty | 67.07 | 67.07 | 0.00 | 841.00 | landing_shaping=2.049 progress_reward=0.206 stability_penalty=-0.079 | new_best |
| 5 | landing_shaping + progress_reward + stability_penalty | -113.70 | 67.07 | -180.78 | 72.00 | landing_shaping=0.011 progress_reward=1.291 stability_penalty=-0.238 | no_meaningful_improvement |

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
- distance_anchor can hurt performance when combined with strong progress_reward
- stability_penalty should not dominate total reward; keep mean magnitude below 0.2
- landing_shaping conditions should be relaxed to increase nonzero rate above 0.1
- progress_reward coefficient may need to be increased beyond 50 to achieve meaningful progress

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| landing_shaping | 0.010606 | 0.010606 | 0.007734 | 0.000000 | 2.392725 |
| progress_reward | 1.290767 | 1.364724 | 0.999991 | -3.195675 | 3.387974 |
| stability_penalty | -0.237851 | 0.237851 | 1.000000 | -2.115911 | -0.000000 |
| total_reward | 1.063521 | 1.181778 | 1.000000 | -4.285132 | 3.004780 |
| generated_reward | 1.063521 | 1.181778 | 1.000000 | -4.285132 | 3.004780 |
| original_env_reward | -1.636045 | 2.433724 | 1.000000 | -100.000000 | 133.146437 |
