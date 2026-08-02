# Training Feedback

## Final-policy outcome
score=-37.479409, len=370.400000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-113.406638, 100.932073]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 115.031601 | 88.3% | 96.4% | 99.6% |
| stability_angle_penalty | -4.620601 | -3.5% | 3.5% | 11.3% |
| ground_penalty | -0.120000 | -0.1% | 0.1% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
