# Training Feedback

## Final-policy outcome
score=-24.239698, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-49.710011, 11.477662]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 265.024510 | 54.5% | 54.5% | 65.8% |
| proximity_reward | 205.725792 | 42.3% | 42.3% | 100.0% |
| progress_reward | 12.171009 | 2.5% | 2.5% | 83.7% |
| stability_penalty | -3.102215 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
