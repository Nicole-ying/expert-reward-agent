# Training Feedback

## Final-policy outcome
score=-37.126816, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-902.563877, 150.464044]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward | 2577.877857 | 68.8% | 69.3% | 93.2% |
| height_gate | 944.245138 | 25.2% | 25.2% | 100.0% |
| lateral_penalty | -109.586332 | -2.9% | 2.9% | 93.1% |
| upright_reward | 78.341314 | 2.1% | 2.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
