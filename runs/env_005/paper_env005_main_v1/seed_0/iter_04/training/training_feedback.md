# Training Feedback

## Final-policy outcome
score=-49.532810, len=819.400000, terminated=6/20, truncated=14/20, reward_errors=0
score_range=[-161.983951, 58.819689]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity_reward | 850.561421 | 59.2% | 59.6% | 99.8% |
| orientation_reward | 394.316276 | 27.4% | 27.4% | 100.0% |
| stability_penalty | -185.760147 | -12.9% | 12.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
