# Training Feedback

## Final-policy outcome
score=250.953879, len=333.850000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[226.153712, 275.934471]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 12.878966 | 74.1% | 76.0% | 95.9% |
| angle_penalty | -2.880552 | -16.6% | 16.6% | 100.0% |
| landing_reward | 1.283246 | 7.4% | 7.4% | 97.5% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
