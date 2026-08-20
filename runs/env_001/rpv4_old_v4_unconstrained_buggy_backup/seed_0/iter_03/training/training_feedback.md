# Training Feedback

## Final-policy outcome
score=-112.673655, len=68.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335058, 1.552658]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vel_damping | -19.975020 | -53.9% | 53.9% | 100.0% |
| landing_bonus | 12.439211 | 33.6% | 33.6% | 2.2% |
| progress_reward | 3.357916 | 9.1% | 9.4% | 100.0% |
| orientation_penalty | -0.964653 | -2.6% | 2.6% | 100.0% |
| action_penalty | -0.207500 | -0.6% | 0.6% | 6.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 19/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
