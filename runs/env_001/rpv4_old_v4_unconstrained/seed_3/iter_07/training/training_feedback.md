# Training Feedback

## Final-policy outcome
score=24.815189, len=687.850000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-267.150515, 262.620642]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settled_reward | 326.066588 | 76.3% | 76.3% | 11.1% |
| goal_proximity | -57.383571 | -13.4% | 13.4% | 100.0% |
| contact_bonus | 24.360957 | 5.7% | 5.7% | 0.7% |
| landing_approach | 8.433396 | 2.0% | 2.0% | 100.0% |
| velocity_penalty | -5.138310 | -1.2% | 1.2% | 99.7% |
| fuel_penalty | -2.854000 | -0.7% | 0.7% | 83.0% |
| orientation_penalty | -1.981543 | -0.5% | 0.5% | 100.0% |
| contact_speed_penalty | -1.345398 | -0.3% | 0.3% | 10.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
