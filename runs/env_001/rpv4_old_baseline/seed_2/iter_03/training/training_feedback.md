# Training Feedback

## Final-policy outcome
score=-80.086225, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-104.209449, -46.051412]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 983.148105 | 36.9% | 36.9% | 100.0% |
| angle_reward | 645.188928 | 24.2% | 24.2% | 100.0% |
| speed_reward | 533.346849 | 20.0% | 20.0% | 100.0% |
| height_reward | 502.390004 | 18.9% | 18.9% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
