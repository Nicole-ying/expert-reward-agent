# Training Feedback

## Final-policy outcome
score=-65.961915, len=1600.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-99.664145, -52.310843]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| alternating_bonus | 144.335000 | 79.1% | 79.1% | 90.2% |
| both_off_ground_penalty | -21.400000 | -11.7% | 11.7% | 6.7% |
| balance_penalty | -9.211715 | -5.0% | 5.0% | 100.0% |
| forward_progress_reward | 7.553309 | 4.1% | 4.1% | 84.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
