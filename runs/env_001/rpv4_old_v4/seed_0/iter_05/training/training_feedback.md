# Training Feedback

## Final-policy outcome
score=-124.216396, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335058, -105.319405]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vel_damping | -11.707847 | -70.5% | 70.5% | 45.6% |
| landing_proximity_reward | 2.316456 | 13.9% | 13.9% | 100.0% |
| orientation_penalty | -1.330045 | -8.0% | 8.0% | 100.0% |
| progress_reward | 1.114200 | 6.7% | 6.9% | 100.0% |
| fuel_penalty | -0.110000 | -0.7% | 0.7% | 3.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
