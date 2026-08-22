# Training Feedback

## Final-policy outcome
score=-98.930747, len=69.600000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-142.817091, -60.484766]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_shaping | 5.553016 | 49.7% | 51.7% | 100.0% |
| success_landing_bonus | 5.000000 | 44.7% | 44.7% | 0.1% |
| angle_penalty | -0.220921 | -2.0% | 2.0% | 100.0% |
| action_cost | -0.178000 | -1.6% | 1.6% | 12.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
