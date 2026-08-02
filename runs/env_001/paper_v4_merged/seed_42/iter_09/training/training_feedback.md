# Training Feedback

## Final-policy outcome
score=-110.682953, len=69.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-205.796452, 12.522001]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_delta | 1.118792 | 69.9% | 72.4% | 100.0% |
| success_proxy | 0.273736 | 17.1% | 17.1% | 0.5% |
| fuel_penalty | -0.167500 | -10.5% | 10.5% | 4.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 18/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
