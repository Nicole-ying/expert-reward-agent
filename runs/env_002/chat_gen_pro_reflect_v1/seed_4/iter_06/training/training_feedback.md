# Training Feedback

## Final-policy outcome
score=-59.502847, len=136.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-66.980569, -45.089544]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 201.808555 | 48.9% | 48.9% | 99.3% |
| gated_forward | 124.201073 | 30.1% | 30.1% | 98.8% |
| forward_reward | 83.344789 | 20.2% | 20.2% | 99.5% |
| energy_penalty | -3.022207 | -0.7% | 0.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 16/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
