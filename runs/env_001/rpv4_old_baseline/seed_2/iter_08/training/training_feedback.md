# Training Feedback

## Final-policy outcome
score=-68.154281, len=712.800000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-224.490992, 157.404283]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_reward | 360.234230 | 81.7% | 81.7% | 5.5% |
| fuel_penalty | -35.402500 | -8.0% | 8.0% | 99.3% |
| proximity_penalty | -31.232415 | -7.1% | 7.1% | 100.0% |
| time_penalty | -7.128000 | -1.6% | 1.6% | 100.0% |
| descent_reward | 7.122844 | 1.6% | 1.6% | 89.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
