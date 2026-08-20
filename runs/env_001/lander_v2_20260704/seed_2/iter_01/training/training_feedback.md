# Training Feedback

## Final-policy outcome
score=-110.971651, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-127.611244, -95.909090]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_reward | -66.465033 | -88.7% | 88.7% | 100.0% |
| stability_penalty | -8.304364 | -11.1% | 11.1% | 100.0% |
| soft_landing_bonus | 0.175000 | 0.2% | 0.2% | 0.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
