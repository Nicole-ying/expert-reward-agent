# Training Feedback

## Final-policy outcome
score=-216.087920, len=152.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-275.446791, -159.313481]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vel_penalty | -2.568269 | -47.6% | 47.6% | 100.0% |
| progress_reward | -0.130336 | -2.4% | 26.4% | 100.0% |
| fuel_penalty | -1.358750 | -25.2% | 25.2% | 47.2% |
| orientation_penalty | -0.044856 | -0.8% | 0.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 10/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
