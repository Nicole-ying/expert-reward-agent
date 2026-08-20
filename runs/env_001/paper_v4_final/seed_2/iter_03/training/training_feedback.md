# Training Feedback

## Final-policy outcome
score=222.690009, len=547.700000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[112.761295, 275.006661]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 426.356525 | 94.4% | 94.4% | 100.0% |
| pose_penalty | -15.447610 | -3.4% | 3.4% | 100.0% |
| progress | -10.022957 | -2.2% | 2.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
