# Training Feedback

## Final-policy outcome
score=278.882215, len=1077.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-1.524912, 297.530390]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 488.806239 | 97.3% | 97.5% | 100.0% |
| balance_penalty | -12.352668 | -2.5% | 2.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
