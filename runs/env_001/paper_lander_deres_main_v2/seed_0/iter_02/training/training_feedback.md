# Training Feedback

## Final-policy outcome
score=-566.799104, len=114.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-643.085948, -479.902330]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_delta | -0.126739 | -19.2% | 74.7% | 100.0% |
| stability_penalty | -0.167199 | -25.3% | 25.3% | 26.4% |
| soft_landing_proxy | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 17/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
