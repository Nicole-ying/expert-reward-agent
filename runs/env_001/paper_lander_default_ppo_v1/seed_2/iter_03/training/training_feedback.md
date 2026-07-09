# Training Feedback

## Final-policy outcome
score=-98.507232, len=68.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-122.801170, -77.628538]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_progress | 2.241351 | 67.9% | 70.4% | 100.0% |
| stability_penalty | -0.777627 | -23.6% | 23.6% | 100.0% |
| soft_landing_proxy | 0.200000 | 6.1% | 6.1% | 0.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
