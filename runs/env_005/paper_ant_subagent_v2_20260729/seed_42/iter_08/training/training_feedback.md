# Training Feedback

## Final-policy outcome
score=-845.860418, len=796.900000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[-1762.000653, -68.179497]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 668.367557 | 69.8% | 80.7% | 63.2% |
| lateral_penalty | -71.276449 | -7.4% | 7.4% | 43.8% |
| roll_penalty | -55.106619 | -5.8% | 5.8% | 42.5% |
| pitch_penalty | -48.741792 | -5.1% | 5.1% | 39.2% |
| height_low_penalty | -9.249895 | -1.0% | 1.0% | 28.9% |
| height_high_penalty | -0.198375 | -0.0% | 0.0% | 0.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
