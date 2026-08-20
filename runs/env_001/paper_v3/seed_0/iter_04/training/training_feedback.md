# Training Feedback

## Final-policy outcome
score=-111.561339, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-127.324982, -95.059093]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_penalty | -66.496863 | -78.2% | 78.2% | 100.0% |
| velocity_penalty | -11.447106 | -13.5% | 13.5% | 99.6% |
| success_bonus | 5.000000 | 5.9% | 5.9% | 0.1% |
| landing_proxy | 1.667363 | 2.0% | 2.0% | 0.9% |
| orientation_penalty | -0.476705 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
