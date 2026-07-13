# Training Feedback

## Final-policy outcome
score=0.161808, len=963.100000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-55.725850, 201.909333]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 655.191869 | 97.9% | 97.9% | 100.0% |
| progress_reward | 5.843411 | 0.9% | 1.2% | 99.9% |
| stability_penalty | -5.953156 | -0.9% | 0.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
