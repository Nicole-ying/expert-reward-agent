# Iteration Context

## Recommended Action
**tune** — Current score -101.17 is best so far, but still far from target 200. Progress_reward coefficient 10.0 is weak; stability_penalty dominates. Landing_shaping rarely activates. Tune: increase progress_reward coefficient to 50.0, reduce stability_penalty weights, and relax landing_shaping conditions to increase activation.

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_bonus + stability_penalty | -108.58 | -108.58 | 0.00 | 72.00 | progress_reward=0.160 soft_landing_bonus=0.012 stability_penalty=-0.342 | new_best |
| 2 | distance_anchor + landing_shaping + progress_reward + stability_penalty | -112.24 | -108.58 | -3.66 | 72.00 | distance_anchor=-0.485 landing_shaping=0.012 progress_reward=0.802 stability_penalty=-0.406 | no_meaningful_improvement |
| 3 | landing_shaping + progress_reward + stability_penalty | -101.17 | -101.17 | 0.00 | 72.00 | landing_shaping=0.013 progress_reward=0.160 stability_penalty=-0.338 | new_best |

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

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| landing_shaping | 0.013000 | 0.013000 | 0.007690 | 0.000000 | 2.817319 |
| progress_reward | 0.159678 | 0.169155 | 0.999986 | -0.410798 | 0.420486 |
| stability_penalty | -0.338351 | 0.338351 | 1.000000 | -2.933820 | -0.000000 |
| total_reward | -0.165672 | 0.190218 | 1.000000 | -2.815020 | 2.814434 |
| generated_reward | -0.165672 | 0.190218 | 1.000000 | -2.815020 | 2.814434 |
| original_env_reward | -1.522998 | 2.415771 | 1.000000 | -100.000000 | 121.431447 |
