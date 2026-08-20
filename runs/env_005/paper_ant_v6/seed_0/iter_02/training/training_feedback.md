# Training Feedback

## Final-policy outcome
score=-14.407533, len=938.850000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-382.488862, 71.997505]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 1191.718771 | 94.1% | 94.1% | 96.4% |
| lateral_penalty | -74.461135 | -5.9% | 5.9% | 99.5% |
| height_penalty | -0.021792 | -0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
