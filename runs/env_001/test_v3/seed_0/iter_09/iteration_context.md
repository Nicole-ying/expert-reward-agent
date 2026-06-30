# Iteration Context

## Recommended Action
**rebuild** — 当前骨架已迭代4轮，最佳得分158.82未达到目标200，且最近一轮得分100.05低于最佳。progress_reward系数50已较高但得分仍不足，landing_shaping触发率48%但未带来突破。历史中该骨架多次出现得分波动，表明结构瓶颈。建议重建骨架，考虑引入新的密集信号或调整组件角色。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_bonus + stability_penalty | -111.26 | -111.26 | 0.00 | 74.10 | progress_reward=0.160 soft_landing_bonus=0.011 stability_penalty=-0.340 | new_best |
| 2 | progress_reward + soft_landing_bonus + stability_penalty | -111.79 | -111.26 | -0.53 | 74.10 | progress_reward=0.242 soft_landing_bonus=0.010 stability_penalty=-0.331 | no_meaningful_improvement |
| 3 | landing_shaping + progress_reward + stability_penalty | 158.82 | 158.82 | 0.00 | 728.40 | landing_shaping=1.625 progress_reward=0.204 stability_penalty=-0.037 | new_best |
| 4 | landing_shaping + progress_reward + stability_penalty | -142.80 | 158.82 | -301.62 | 88.10 | landing_shaping=0.012 progress_reward=1.599 stability_penalty=-0.122 | no_meaningful_improvement |
| 5 | landing_shaping + progress_reward + stability_penalty | -190.67 | 158.82 | -349.50 | 84.30 | landing_shaping=0.010 progress_reward=5.119 stability_penalty=-0.068 | no_meaningful_improvement |
| 6 | distance_reward + progress_reward + stability_penalty | -109.08 | 158.82 | -267.91 | 83.60 | distance_reward=-0.093 progress_reward=8.255 stability_penalty=-0.133 | unsolved_stagnation_fresh_restart |
| 7 | progress_reward + soft_landing_bonus + stability_penalty | 146.57 | 158.82 | -12.25 | 564.60 | progress_reward=0.091 soft_landing_bonus=0.397 stability_penalty=-0.146 | no_meaningful_improvement |
| 8 | landing_shaping + progress_reward + stability_penalty | 100.05 | 158.82 | -58.77 | 846.40 | landing_shaping=1.370 progress_reward=0.228 stability_penalty=-0.022 | no_meaningful_improvement |

## Expert Cards
## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## Training Evidence
# Training Feedback

## External evaluation
- score: 100.050499
- episode_length: 846.400000 (mean)
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| landing_shaping | 1.369955 | 1.369955 | 0.481882 | 0.000000 | 4.990918 |
| progress_reward | 0.227725 | 0.245766 | 0.999402 | -1.641093 | 2.271038 |
| stability_penalty | -0.021996 | 0.021996 | 1.000000 | -0.273139 | -0.000000 |
| total_reward | 1.575683 | 1.593989 | 1.000000 | -1.742662 | 4.990927 |
| generated_reward | 1.575683 | 1.593989 | 1.000000 | -1.742662 | 4.990927 |
| original_env_reward | -0.301860 | 2.052598 | 1.000000 | -100.000000 | 139.497487 |

## Signals
partial_progress; penalty_dominance:landing_shaping; penalty_dominance:generated_reward; penalty_dominance:original_env_reward
