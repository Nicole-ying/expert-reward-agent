# Training Feedback

## Final-policy outcome
score=267.803042, len=925.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[1.705430, 317.980675]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_gated | 469.242136 | 67.0% | 67.0% | 100.0% |
| stability_bonus | 210.296992 | 30.0% | 30.0% | 100.0% |
| action_efficiency | -20.810552 | -3.0% | 3.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
