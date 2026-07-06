# Training Feedback

## Final-policy outcome
score=380.749953, len=234.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[360.663059, 395.033875]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_stability_reward | 136.630464 | 91.0% | 92.0% | 100.0% |
| vertical_pushoff | 11.139575 | 7.4% | 7.4% | 40.7% |
| stability_penalty | -0.811615 | -0.5% | 0.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
