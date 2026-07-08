# Training Feedback

## Final-policy outcome
score=-115.438926, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-137.036041, -94.040470]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 2.500000 | 51.8% | 51.8% | 0.4% |
| stability_penalty | -1.164824 | -24.1% | 24.1% | 100.0% |
| progress_delta | 1.121694 | 23.2% | 24.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
