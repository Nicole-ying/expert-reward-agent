# Training Feedback

## Final-policy outcome
score=189.341696, len=1348.350000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[-0.172268, 292.870681]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 378.735037 | 71.2% | 71.2% | 100.0% |
| stability_cost | -141.989323 | -26.7% | 26.7% | 100.0% |
| action_cost | -10.994526 | -2.1% | 2.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
