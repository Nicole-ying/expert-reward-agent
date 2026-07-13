# Training Feedback

## Final-policy outcome
score=-85.225247, len=69.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-117.831261, -56.370675]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 5.590425 | 85.2% | 88.4% | 100.0% |
| landing_quality | 0.751702 | 11.5% | 11.5% | 3.5% |
| orientation_penalty | -0.008576 | -0.1% | 0.1% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
