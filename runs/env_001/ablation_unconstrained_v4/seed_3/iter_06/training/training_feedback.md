# Training Feedback

## Final-policy outcome
score=-122.772028, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-146.655919, -101.246762]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 7.000000 | 68.9% | 68.9% | 0.5% |
| lateral_penalty | -2.105754 | -20.7% | 20.7% | 100.0% |
| progress_reward | 0.802204 | 7.9% | 8.4% | 100.0% |
| engine_penalty | -0.202500 | -2.0% | 2.0% | 5.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
