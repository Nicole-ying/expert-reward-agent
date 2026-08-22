# Training Feedback

## Final-policy outcome
score=-120.095465, len=76.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-183.306848, -7.740426]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 42.500000 | 91.0% | 91.0% | 1.1% |
| descent_reward | 2.868708 | 6.1% | 6.1% | 86.1% |
| progress_reward | 1.037989 | 2.2% | 2.6% | 100.0% |
| action_penalty | -0.087000 | -0.2% | 0.2% | 11.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 17/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
