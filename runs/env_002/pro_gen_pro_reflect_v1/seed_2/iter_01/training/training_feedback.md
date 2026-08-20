# Training Feedback

## Final-policy outcome
score=290.575870, len=930.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[288.840362, 292.423217]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 504.315988 | 98.9% | 99.0% | 100.0% |
| stability_penalty | -2.994604 | -0.6% | 0.6% | 100.0% |
| energy_penalty | -2.227004 | -0.4% | 0.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
