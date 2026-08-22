# Training Feedback

## Final-policy outcome
score=266.034262, len=284.000000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[236.701754, 299.578339]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 27.502953 | 56.1% | 57.2% | 94.9% |
| landing_quality | 11.754354 | 24.0% | 24.0% | 14.3% |
| engine_cost | -4.835000 | -9.9% | 9.9% | 85.1% |
| attitude_penalty | -4.111881 | -8.4% | 8.4% | 100.0% |
| landing_velocity_penalty | -0.305580 | -0.6% | 0.6% | 10.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
