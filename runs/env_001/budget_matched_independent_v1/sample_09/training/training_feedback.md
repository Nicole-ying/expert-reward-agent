# Training Feedback

## Final-policy outcome
score=33.076228, len=934.700000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-33.167327, 188.299497]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 21.264301 | 88.1% | 88.1% | 1.5% |
| velocity_penalty | -1.475075 | -6.1% | 6.1% | 99.7% |
| progress | 1.370560 | 5.7% | 5.7% | 72.6% |
| angular_penalty | -0.021955 | -0.1% | 0.1% | 99.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
