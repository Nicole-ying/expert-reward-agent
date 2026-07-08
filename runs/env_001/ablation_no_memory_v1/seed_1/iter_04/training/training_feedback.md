# Training Feedback

## Final-policy outcome
score=-110.808086, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-129.407665, -95.059093]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_penalty | -4.454561 | -41.6% | 41.6% | 100.0% |
| progress_delta | 3.362381 | 31.4% | 32.5% | 100.0% |
| deceleration_bonus | 2.777548 | 25.9% | 25.9% | 8.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
