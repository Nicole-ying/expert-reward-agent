# Iteration Context

## Recommended Action
**tune** — Current score -108.58 is far below target 200. Progress reward coefficient (10.0) is too weak; stability penalty dominates. Best_reward.py is identical to previous, so no revert needed. Increase progress coefficient to at least 50, reduce stability penalty weights, and consider adding a distance anchor to prevent oscillation.

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_bonus + stability_penalty | -108.58 | -108.58 | 0.00 | 72.00 | progress_reward=0.160 soft_landing_bonus=0.012 stability_penalty=-0.342 | new_best |

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
| progress_reward | 0.159545 | 0.168953 | 0.999991 | -0.409497 | 0.424014 |
| soft_landing_bonus | 0.012486 | 0.012486 | 0.006243 | 0.000000 | 2.000000 |
| stability_penalty | -0.342410 | 0.342410 | 1.000000 | -2.824370 | -0.000001 |
| total_reward | -0.170379 | 0.193710 | 1.000000 | -2.985490 | 2.000806 |
| generated_reward | -0.170379 | 0.193710 | 1.000000 | -2.985490 | 2.000806 |
| original_env_reward | -1.522908 | 2.386149 | 1.000000 | -100.000000 | 129.280614 |
