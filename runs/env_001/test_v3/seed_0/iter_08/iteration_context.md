# Iteration Context

## Recommended Action
**rebuild** — 当前骨架已迭代3轮，最佳得分146.57未达到目标200，且低于历史最佳158.82。progress_reward系数10.0过弱（均值0.091），stability_penalty系数过高（均值-0.146）主导总奖励，soft_landing_bonus稀疏（触发率19.8%）。最佳奖励使用progress_reward系数50.0、stability_penalty更低系数、连续landing_shaping，得分更高。建议回退到最佳奖励骨架并微调。

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

## Expert Cards
## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## sparse_completion_proxy
- signal: completion/landing bonus trigger_rate < 1%
- risk: final bonus provides little early learning guidance
- fix: replace hard bonus with smoother landing-quality shaping

## Training Evidence
# Training Feedback

## External evaluation
- score: 146.572765
- episode_length: 564.600000 (mean)
- range: [-69.127123, 295.110743]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| progress_reward | 0.091198 | 0.100075 | 0.999585 | -0.354730 | 0.384918 |
| soft_landing_bonus | 0.396895 | 0.396895 | 0.198447 | 0.000000 | 2.000000 |
| stability_penalty | -0.145633 | 0.145633 | 1.000000 | -8.460213 | -0.000000 |
| total_reward | 0.342459 | 0.452185 | 1.000000 | -8.304337 | 2.004369 |
| generated_reward | 0.342459 | 0.452185 | 1.000000 | -8.304337 | 2.004369 |
| original_env_reward | -0.388388 | 2.619071 | 1.000000 | -100.000000 | 143.501626 |

## Signals
partial_progress; penalty_dominance:soft_landing_bonus; penalty_dominance:stability_penalty; penalty_dominance:generated_reward; penalty_dominance:original_env_reward
