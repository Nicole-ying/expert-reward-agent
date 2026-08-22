# Training Feedback

## Final-policy outcome
score=107.313665, len=828.200000, terminated=6/20, truncated=14/20, reward_errors=0
score_range=[32.903192, 258.358486]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 1323.714629 | 98.0% | 98.0% | 9.5% |
| progress_reward | 20.212194 | 1.5% | 1.7% | 99.8% |
| attitude_penalty | -3.312610 | -0.2% | 0.2% | 100.0% |
| crash_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
