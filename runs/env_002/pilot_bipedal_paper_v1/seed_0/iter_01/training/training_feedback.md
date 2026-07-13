# Training Feedback

## Final-policy outcome
score=150.577729, len=756.000000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-112.406633, 291.032489]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 326.824416 | 89.0% | 90.0% | 100.0% |
| stability_cost | -36.684409 | -10.0% | 10.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 7/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
