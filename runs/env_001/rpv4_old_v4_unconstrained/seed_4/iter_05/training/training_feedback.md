# Training Feedback

## Final-policy outcome
score=-98.299635, len=68.850000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-118.010880, -68.230143]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | -38.574747 | -95.4% | 95.4% | 100.0% |
| velocity_penalty | -1.829619 | -4.5% | 4.5% | 100.0% |
| angle_penalty | -0.030607 | -0.1% | 0.1% | 100.0% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
