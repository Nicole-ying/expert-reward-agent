# Training Feedback

## Final-policy outcome
score=198.143841, len=406.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[29.788218, 259.365378]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_proxy | 82.787236 | 97.8% | 97.8% | 16.9% |
| approach_reward | 1.116209 | 1.3% | 1.5% | 97.3% |
| stability_penalty | -0.540050 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
