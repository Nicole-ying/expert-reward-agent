# Training Feedback

## Final-policy outcome
score=10.311966, len=788.250000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-66.990115, 192.251014]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_approach | 27.816165 | 48.5% | 48.5% | 100.0% |
| goal_proximity | -19.561549 | -34.1% | 34.1% | 100.0% |
| fuel_penalty | -4.677300 | -8.2% | 8.2% | 98.9% |
| velocity_penalty | -2.945820 | -5.1% | 5.1% | 100.0% |
| orientation_penalty | -1.860451 | -3.2% | 3.2% | 100.0% |
| settled_reward | 0.254703 | 0.4% | 0.4% | 2.5% |
| contact_speed_penalty | -0.180001 | -0.3% | 0.3% | 2.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
