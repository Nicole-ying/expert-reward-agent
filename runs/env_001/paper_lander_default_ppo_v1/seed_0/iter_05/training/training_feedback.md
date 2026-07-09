# Training Feedback

## Final-policy outcome
score=-47.906831, len=85.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-123.747614, 29.859505]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_shaping_reward | 36.212216 | 78.8% | 93.7% | 100.0% |
| landing_contact_reward | 1.500000 | 3.3% | 3.3% | 0.4% |
| progress_reward | 1.245254 | 2.7% | 2.8% | 100.0% |
| stability_penalty | -0.106330 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 8/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
