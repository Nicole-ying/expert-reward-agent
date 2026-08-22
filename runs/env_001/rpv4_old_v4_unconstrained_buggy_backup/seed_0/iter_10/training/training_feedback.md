# Training Feedback

## Final-policy outcome
score=-111.923038, len=68.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335058, 16.439465]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 13.423284 | 40.6% | 42.0% | 100.0% |
| landing_bonus | 10.751132 | 32.5% | 32.5% | 2.1% |
| vel_damping | -7.169021 | -21.7% | 21.7% | 100.0% |
| orientation_penalty | -1.118467 | -3.4% | 3.4% | 100.0% |
| action_penalty | -0.150000 | -0.5% | 0.5% | 4.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 19/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
