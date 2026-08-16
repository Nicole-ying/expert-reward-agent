# Training Feedback

## Final-policy outcome
score=-168.021945, len=387.200000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-264.295058, -124.453488]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | -0.071833 | -36.4% | 93.7% | 100.0% |
| orientation_penalty | -0.012515 | -6.3% | 6.3% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| vel_damping | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
