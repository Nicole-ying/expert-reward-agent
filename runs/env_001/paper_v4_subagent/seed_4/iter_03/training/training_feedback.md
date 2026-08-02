# Training Feedback

## Final-policy outcome
score=-61.639116, len=72.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-98.533973, -16.940142]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_reward | 10.000000 | 65.3% | 65.3% | 0.1% |
| progress | 3.263310 | 21.3% | 22.4% | 100.0% |
| action_penalty | -1.057500 | -6.9% | 6.9% | 29.1% |
| stability_penalty | -0.786637 | -5.1% | 5.1% | 100.0% |
| soft_landing | 0.046396 | 0.3% | 0.3% | 0.3% |
| failure_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 12/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
