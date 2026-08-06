# Training Feedback

## Final-policy outcome
score=-122.830710, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-147.719906, -99.206220]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 201.363514 | 45.1% | 54.9% | 3.0% |
| radial_reward | 133.090750 | 29.8% | 31.8% | 100.0% |
| vel_penalty | -38.642529 | -8.6% | 8.6% | 100.0% |
| angle_penalty | -11.693155 | -2.6% | 2.6% | 100.0% |
| time_penalty | -3.415000 | -0.8% | 0.8% | 100.0% |
| descent_reward | 2.864226 | 0.6% | 0.6% | 94.3% |
| proximity_bonus | 2.135681 | 0.5% | 0.5% | 20.7% |
| engine_penalty | -0.580000 | -0.1% | 0.1% | 4.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
