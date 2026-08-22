# Training Feedback

## Final-policy outcome
score=96.283534, len=682.100000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[-86.101801, 216.148604]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 35.622280 | 56.1% | 56.1% | 7.9% |
| engine_cost | -12.477000 | -19.7% | 19.7% | 91.5% |
| progress | 4.902779 | 7.7% | 11.6% | 99.1% |
| attitude_penalty | -5.598229 | -8.8% | 8.8% | 100.0% |
| landing_velocity_penalty | -2.392534 | -3.8% | 3.8% | 17.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
