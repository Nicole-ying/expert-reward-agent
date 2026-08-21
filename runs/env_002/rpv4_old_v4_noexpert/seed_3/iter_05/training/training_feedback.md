# Training Feedback

## Final-policy outcome
score=180.281918, len=1141.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-81.068275, 262.138134]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 606.416379 | 91.9% | 92.0% | 100.0% |
| balance_penalty | -52.835982 | -8.0% | 8.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
