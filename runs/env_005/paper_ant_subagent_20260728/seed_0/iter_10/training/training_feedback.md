# Training Feedback

## Final-policy outcome
score=-292.467154, len=785.650000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[-1512.464338, 46.663934]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 719.382783 | 43.0% | 45.4% | 95.9% |
| _height_gate | 702.714020 | 42.0% | 42.0% | 99.9% |
| upright_bonus | 134.895110 | 8.1% | 8.1% | 88.1% |
| lateral_penalty | -72.515449 | -4.3% | 4.3% | 93.3% |
| action_penalty | -2.244195 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
