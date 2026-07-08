# Training Feedback

## Final-policy outcome
score=-107.397144, len=68.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-123.148677, -84.826744]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 22.395811 | 83.8% | 86.7% | 100.0% |
| landing_proxy | 2.295649 | 8.6% | 8.6% | 0.9% |
| stability_penalty | -1.253889 | -4.7% | 4.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
