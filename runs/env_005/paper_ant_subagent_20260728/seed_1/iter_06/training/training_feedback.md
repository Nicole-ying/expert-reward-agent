# Training Feedback

## Final-policy outcome
score=-231.683034, len=320.350000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-1129.435872, 31.720068]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward | 380.264009 | 50.1% | 56.3% | 100.0% |
| height_factor | 319.258559 | 42.0% | 42.0% | 100.0% |
| lateral_penalty | -7.940552 | -1.0% | 1.0% | 41.0% |
| vertical_penalty | -4.514129 | -0.6% | 0.6% | 27.4% |
| upright_penalty | -0.011902 | -0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
