# Training Feedback

## Final-policy outcome
score=2.304401, len=506.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-201.721652, 224.812115]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 148.000000 | 87.5% | 87.5% | 5.8% |
| progress | 7.763636 | 4.6% | 8.8% | 99.8% |
| angle_penalty | -4.608957 | -2.7% | 2.7% | 100.0% |
| landing_reward | 1.650446 | 1.0% | 1.0% | 98.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
