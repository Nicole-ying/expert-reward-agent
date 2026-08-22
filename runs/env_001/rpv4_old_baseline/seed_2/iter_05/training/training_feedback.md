# Training Feedback

## Final-policy outcome
score=-72.180205, len=77.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-128.384168, 15.106691]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 60.000000 | 71.0% | 71.0% | 0.4% |
| contact_reward | 9.700000 | 11.5% | 11.5% | 4.1% |
| vel_penalty | -9.097190 | -10.8% | 10.8% | 100.0% |
| horiz_penalty | -2.744424 | -3.2% | 3.2% | 100.0% |
| descent_reward | 1.140779 | 1.3% | 1.4% | 100.0% |
| time_penalty | -0.771000 | -0.9% | 0.9% | 100.0% |
| orient_penalty | -0.737094 | -0.9% | 0.9% | 100.0% |
| angvel_penalty | -0.303297 | -0.4% | 0.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 15/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
