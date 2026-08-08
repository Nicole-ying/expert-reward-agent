# Training Feedback

## Final-policy outcome
score=-67.143830, len=256.400000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-89.220574, -38.093204]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_gate | 245.042124 | 62.4% | 62.4% | 99.9% |
| forward_reward | 66.048404 | 16.8% | 16.8% | 83.2% |
| gated_forward | 64.142264 | 16.3% | 16.3% | 83.1% |
| contact_transition_reward | 14.950065 | 3.8% | 3.8% | 100.0% |
| energy_penalty | -2.391720 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 6/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
