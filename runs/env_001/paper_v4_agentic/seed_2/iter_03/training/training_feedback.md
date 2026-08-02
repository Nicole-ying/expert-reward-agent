# Training Feedback

## Final-policy outcome
score=-114.965203, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-140.885582, -96.572743]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_delta | 55.937931 | 83.2% | 86.1% | 100.0% |
| velocity_danger | -8.541283 | -12.7% | 12.7% | 100.0% |
| orientation_penalty | -0.820410 | -1.2% | 1.2% | 100.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
