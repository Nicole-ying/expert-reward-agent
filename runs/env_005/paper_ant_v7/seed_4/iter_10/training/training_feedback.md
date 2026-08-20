# Training Feedback

## Final-policy outcome
score=-169.497967, len=839.350000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-1788.159131, 420.647862]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 1065.061228 | 83.7% | 85.1% | 94.8% |
| angular_penalty | -140.508974 | -11.0% | 11.0% | 78.5% |
| lateral_penalty | -49.214188 | -3.9% | 3.9% | 81.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
