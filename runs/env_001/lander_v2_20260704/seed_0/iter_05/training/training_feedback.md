# Training Feedback

## Final-policy outcome
score=-23.315026, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-66.394917, 15.177975]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 484.418257 | 98.2% | 98.2% | 79.3% |
| distance_reward | 5.890285 | 1.2% | 1.4% | 100.0% |
| stability_penalty | -2.008933 | -0.4% | 0.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
