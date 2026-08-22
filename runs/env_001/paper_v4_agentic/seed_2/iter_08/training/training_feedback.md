# Training Feedback

## Final-policy outcome
score=141.785020, len=955.200000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-37.316795, 188.977091]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| C_landing_steady | 108.323789 | 94.5% | 94.5% | 79.3% |
| A_progress_gated | 5.539801 | 4.8% | 5.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
