# Training Feedback

## Final-policy outcome
score=-6.020912, len=455.450000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-191.187570, 273.476580]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 29.908288 | 82.7% | 82.7% | 25.7% |
| progress | 1.546701 | 4.3% | 9.1% | 100.0% |
| stability_penalty | -1.944877 | -5.4% | 5.4% | 100.0% |
| failure_penalty | -1.000000 | -2.8% | 2.8% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
