# Training Feedback

## Final-policy outcome
score=281.448747, len=905.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[73.399534, 312.282140]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 1705.417594 | 54.7% | 54.7% | 100.0% |
| gated_forward | 904.381548 | 29.0% | 29.0% | 100.0% |
| forward_reward | 480.340274 | 15.4% | 15.4% | 100.0% |
| energy_penalty | -24.941086 | -0.8% | 0.8% | 100.0% |
| angle_hinge_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
