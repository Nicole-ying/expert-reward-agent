# Training Feedback

## Final-policy outcome
score=27.148488, len=404.200000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-883.169692, 442.701101]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 863.896309 | 67.0% | 72.6% | 100.0% |
| upright_penalty | -172.382531 | -13.4% | 13.4% | 100.0% |
| height_reward | 103.395436 | 8.0% | 8.0% | 100.0% |
| action_cost | -77.516854 | -6.0% | 6.0% | 100.0% |
| height_boundary_penalty | -0.305161 | -0.0% | 0.0% | 0.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
