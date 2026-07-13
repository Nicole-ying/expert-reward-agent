# Training Feedback

## Final-policy outcome
score=-119.298281, len=876.500000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-225.757913, -25.729280]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| main_engine_penalty | -10.087000 | -46.4% | 46.4% | 57.5% |
| distance_progress | 2.840111 | 13.1% | 32.5% | 100.0% |
| alive_penalty | -4.382500 | -20.2% | 20.2% | 100.0% |
| angle_penalty | -0.186182 | -0.9% | 0.9% | 100.0% |
| angvel_penalty | -0.020782 | -0.1% | 0.1% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
