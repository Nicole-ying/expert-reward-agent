# Training Feedback

## Final-policy outcome
score=297.676687, len=1032.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[101.258632, 308.601155]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 1956.536958 | 57.3% | 57.3% | 100.0% |
| gated_forward | 939.183820 | 27.5% | 27.5% | 100.0% |
| forward_reward | 495.405691 | 14.5% | 14.5% | 100.0% |
| energy_penalty | -26.007868 | -0.8% | 0.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
