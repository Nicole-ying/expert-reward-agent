# Training Feedback

## Final-policy outcome
score=238.204421, len=397.700000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[140.758031, 271.867349]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 77.950000 | 87.7% | 87.7% | 19.6% |
| target_proximity | -8.682168 | -9.8% | 9.8% | 100.0% |
| velocity_penalty | -1.941557 | -2.2% | 2.2% | 97.1% |
| orientation_penalty | -0.224519 | -0.3% | 0.3% | 100.0% |
| angvel_penalty | -0.120384 | -0.1% | 0.1% | 94.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
