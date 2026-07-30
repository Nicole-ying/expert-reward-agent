# Training Feedback

## Final-policy outcome
score=-591.524103, len=585.900000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-1747.282509, 13.295004]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward | 168.133482 | 50.6% | 63.3% | 81.5% |
| upright_bonus | 73.953752 | 22.3% | 34.1% | 100.0% |
| lateral_penalty | -6.828161 | -2.1% | 2.1% | 23.1% |
| action_penalty | -1.690023 | -0.5% | 0.5% | 100.0% |
| height_penalty | -0.118510 | -0.0% | 0.0% | 20.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
