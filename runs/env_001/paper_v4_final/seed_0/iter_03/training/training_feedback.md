# Training Feedback

## Final-policy outcome
score=-81.431904, len=69.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-108.776750, -45.881498]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 1.120410 | 89.6% | 92.9% | 100.0% |
| angular_velocity_penalty | -0.045280 | -3.6% | 3.6% | 0.9% |
| soft_landing | 0.042450 | 3.4% | 3.4% | 0.7% |
| angle_penalty | -0.000933 | -0.1% | 0.1% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 19/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
