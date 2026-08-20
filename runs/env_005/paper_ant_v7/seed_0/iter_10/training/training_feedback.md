# Training Feedback

## Final-policy outcome
score=10.128395, len=380.650000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-548.162926, 326.019564]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 838.557067 | 77.1% | 81.7% | 100.0% |
| height_reward | 103.841177 | 9.6% | 9.6% | 100.0% |
| action_cost | -78.475518 | -7.2% | 7.2% | 100.0% |
| lateral_velocity_penalty | -16.900497 | -1.6% | 1.6% | 67.7% |
| height_boundary_penalty | -0.237681 | -0.0% | 0.0% | 0.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
