# Training Feedback

## Final-policy outcome
score=-27.074350, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-58.391934, 13.703328]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_damping | -10.677051 | -74.1% | 74.1% | 100.0% |
| orientation | -3.455274 | -24.0% | 24.0% | 100.0% |
| progress | 0.224206 | 1.6% | 1.9% | 100.0% |
| soft_landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
