# Training Feedback

## Final-policy outcome
score=-14.658086, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-42.285507, 13.828694]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing | 472.803964 | 51.7% | 51.7% | 100.0% |
| progress_gated | 318.238222 | 34.8% | 34.8% | 69.7% |
| fuel_penalty | -122.820000 | -13.4% | 13.4% | 61.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
