# Training Feedback

## Final-policy outcome
score=-141.960183, len=68.650000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-219.566644, 36.226115]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| altitude_penalty | -127.476907 | -81.3% | 81.3% | 100.0% |
| contact_reward | 17.000000 | 10.8% | 10.8% | 4.3% |
| distance_progress | 11.314639 | 7.2% | 7.4% | 100.0% |
| angvel_penalty | -0.309023 | -0.2% | 0.2% | 100.0% |
| angle_penalty | -0.273105 | -0.2% | 0.2% | 100.0% |
| fuel_penalty | -0.115000 | -0.1% | 0.1% | 1.7% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 19/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
