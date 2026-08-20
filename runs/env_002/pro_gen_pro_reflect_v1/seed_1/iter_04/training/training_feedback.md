# Training Feedback

## Final-policy outcome
score=276.140978, len=901.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[3.863848, 320.758019]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_gated | 458.334366 | 96.0% | 96.0% | 100.0% |
| action_efficiency | -18.816441 | -3.9% | 3.9% | 100.0% |
| stability_penalty | -0.181695 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
