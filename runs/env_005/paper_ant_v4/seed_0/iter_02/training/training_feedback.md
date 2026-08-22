# Training Feedback

## Final-policy outcome
score=13.818725, len=29.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-39.863111, 35.704146]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_reward | 18.785118 | 34.5% | 52.8% | 96.6% |
| angular_penalty | 11.156708 | 20.5% | 20.5% | 100.0% |
| height_penalty | 9.893848 | 18.2% | 18.2% | 100.0% |
| lateral_penalty | 4.601561 | 8.4% | 8.4% | 100.0% |
| upright_penalty | 0.102423 | 0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
