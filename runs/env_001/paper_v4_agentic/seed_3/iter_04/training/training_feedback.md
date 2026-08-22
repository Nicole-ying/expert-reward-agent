# Training Feedback

## Final-policy outcome
score=-12.980602, len=990.550000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-56.927662, 142.412793]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 1018.503727 | 94.9% | 94.9% | 88.5% |
| progress | 24.223171 | 2.3% | 3.0% | 100.0% |
| engine_cost | -19.795000 | -1.8% | 1.8% | 99.9% |
| attitude_penalty | -2.823626 | -0.3% | 0.3% | 100.0% |
| landing_velocity_penalty | -0.122212 | -0.0% | 0.0% | 0.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
