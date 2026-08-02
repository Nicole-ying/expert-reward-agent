# Training Feedback

## Final-policy outcome
score=141.952632, len=956.300000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[31.857841, 183.791325]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | 39.163109 | 96.3% | 96.3% | 90.5% |
| progress_reward | 1.397127 | 3.4% | 3.6% | 100.0% |
| angle_penalty | -0.057159 | -0.1% | 0.1% | 99.8% |
| angular_velocity_penalty | -0.001900 | -0.0% | 0.0% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
