# Training Feedback

## Final-policy outcome
score=238.929386, len=380.100000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[43.368124, 298.628230]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_proxy | 55.825000 | 60.8% | 60.8% | 29.4% |
| velocity_alignment_reward | 35.490298 | 38.7% | 38.7% | 80.2% |
| stability_penalty | -0.443126 | -0.5% | 0.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
