# Training Feedback

## Final-policy outcome
score=-109.321721, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-126.495599, -90.982986]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 1.120375 | 89.7% | 92.8% | 100.0% |
| soft_landing | 0.043032 | 3.4% | 3.4% | 0.7% |
| angular_velocity_penalty | -0.042957 | -3.4% | 3.4% | 0.7% |
| angle_penalty | -0.003569 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
