# Training Feedback

## Final-policy outcome
score=-585.811974, len=663.700000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-1681.168796, -17.908667]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| survival_bonus | 331.850000 | 58.1% | 58.1% | 100.0% |
| forward_progress | 205.474124 | 36.0% | 41.9% | 47.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
