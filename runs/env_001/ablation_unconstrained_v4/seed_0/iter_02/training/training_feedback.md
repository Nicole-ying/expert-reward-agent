# Training Feedback

## Final-policy outcome
score=108.116443, len=743.300000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[-39.175074, 241.716908]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settle_reward | 1371.660417 | 90.0% | 90.0% | 100.0% |
| distance_penalty | -65.745510 | -4.3% | 4.3% | 100.0% |
| velocity_damping | -51.630633 | -3.4% | 3.4% | 99.9% |
| engine_penalty | -32.910000 | -2.2% | 2.2% | 88.6% |
| progress_delta | 1.240063 | 0.1% | 0.1% | 98.8% |
| orientation_penalty | -0.762181 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
