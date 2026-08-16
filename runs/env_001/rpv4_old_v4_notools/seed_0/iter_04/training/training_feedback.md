# Training Feedback

## Final-policy outcome
score=-159.200781, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-185.223706, -122.233691]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 2084.688992 | 99.9% | 99.9% | 100.0% |
| orientation_penalty | -1.378219 | -0.1% | 0.1% | 100.0% |
| progress_reward | -0.138147 | -0.0% | 0.0% | 100.0% |
| vel_damping | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
