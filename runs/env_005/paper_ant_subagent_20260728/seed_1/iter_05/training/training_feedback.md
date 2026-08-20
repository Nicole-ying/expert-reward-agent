# Training Feedback

## Final-policy outcome
score=-664.527786, len=713.200000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[-1253.279910, -16.270632]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_factor | 609.232254 | 57.0% | 57.0% | 100.0% |
| forward | 136.371126 | 12.8% | 29.2% | 100.0% |
| upright_bonus | 22.893533 | 2.1% | 12.8% | 100.0% |
| lateral_penalty | -6.974759 | -0.7% | 0.7% | 26.7% |
| vertical_penalty | -3.682786 | -0.3% | 0.3% | 11.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
