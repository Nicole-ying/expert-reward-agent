# Iteration Context

## Recommended Action
**tune** — 当前骨架已迭代4轮，最佳得分67.07远低于目标200，且最近两轮无改善。best_reward.py与previous_reward.py代码相同，无需revert。landing_shaping均值2.049过高，可能主导奖励导致agent只关注着陆而忽略进度。progress_reward均值0.206偏弱，需增大系数。建议：增大progress_reward系数至80，降低landing_shaping系数至2.0，并收紧landing条件（dist<=0.5, speed<=0.5）以减少hacking风险。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_bonus + stability_penalty | -108.58 | -108.58 | 0.00 | 72.00 | progress_reward=0.160 soft_landing_bonus=0.012 stability_penalty=-0.342 | new_best |
| 2 | distance_anchor + landing_shaping + progress_reward + stability_penalty | -112.24 | -108.58 | -3.66 | 72.00 | distance_anchor=-0.485 landing_shaping=0.012 progress_reward=0.802 stability_penalty=-0.406 | no_meaningful_improvement |
| 3 | landing_shaping + progress_reward + stability_penalty | -101.17 | -101.17 | 0.00 | 72.00 | landing_shaping=0.013 progress_reward=0.160 stability_penalty=-0.338 | new_best |
| 4 | landing_shaping + progress_reward + stability_penalty | 67.07 | 67.07 | 0.00 | 841.00 | landing_shaping=2.049 progress_reward=0.206 stability_penalty=-0.079 | new_best |
| 5 | landing_shaping + progress_reward + stability_penalty | -113.70 | 67.07 | -180.78 | 72.00 | landing_shaping=0.011 progress_reward=1.291 stability_penalty=-0.238 | no_meaningful_improvement |
| 6 | landing_shaping + progress_reward + stability_penalty | 67.07 | 67.07 | 0.00 | 841.00 | landing_shaping=2.049 progress_reward=0.206 stability_penalty=-0.079 | no_meaningful_improvement |

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
- landing_shaping coefficient should be at least 3.0 to provide meaningful signal
- landing_shaping thresholds should be relaxed (dist<=1.0, speed<=1.0, angle<=0.5) to achieve nonzero rate above 0.1
- progress_reward coefficient of 50 is sufficient; increasing to 80 may cause instability

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| landing_shaping | 2.048909 | 2.048909 | 0.766275 | 0.000000 | 4.994907 |
| progress_reward | 0.205633 | 0.234669 | 0.999400 | -1.502288 | 1.886572 |
| stability_penalty | -0.078701 | 0.078701 | 1.000000 | -1.518036 | -0.000003 |
| total_reward | 2.175842 | 2.203330 | 1.000000 | -2.606938 | 4.994888 |
| generated_reward | 2.175842 | 2.203330 | 1.000000 | -2.606938 | 4.994888 |
| original_env_reward | -0.148962 | 1.868448 | 1.000000 | -100.000000 | 117.805453 |
