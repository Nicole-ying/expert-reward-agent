# Training Feedback

## Final-policy outcome
score=320.799722, len=891.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[318.939377, 322.130437]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_gated | 495.391948 | 96.0% | 96.0% | 100.0% |
| action_penalty | -20.581384 | -4.0% | 4.0% | 100.0% |
| vertical_penalty | -0.130520 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
