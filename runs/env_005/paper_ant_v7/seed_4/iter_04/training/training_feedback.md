# Training Feedback

## Final-policy outcome
score=-779.129977, len=797.250000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-1727.386841, -6.950092]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_gate_value | 522.292814 | 67.6% | 67.6% | 66.1% |
| forward_upright_gated | 143.310715 | 18.5% | 25.1% | 65.1% |
| rotation_penalty | -31.163549 | -4.0% | 4.0% | 44.1% |
| torque_penalty | -12.975266 | -1.7% | 1.7% | 100.0% |
| lateral_penalty | -12.251550 | -1.6% | 1.6% | 36.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
