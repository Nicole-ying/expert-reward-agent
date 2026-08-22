# Training Feedback

## Final-policy outcome
score=-121.737397, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.421811, -99.908123]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 33.779022 | 64.8% | 66.9% | 100.0% |
| landing_bonus | 8.457278 | 16.2% | 16.2% | 1.1% |
| descent_bonus | 7.173807 | 13.8% | 13.8% | 94.3% |
| attitude_penalty | -1.527282 | -2.9% | 2.9% | 100.0% |
| stability_reward | 0.091095 | 0.2% | 0.2% | 1.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
