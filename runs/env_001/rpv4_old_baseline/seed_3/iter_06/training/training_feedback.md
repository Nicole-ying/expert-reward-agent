# Training Feedback

## Final-policy outcome
score=-87.870704, len=773.650000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-152.048351, -30.434026]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| engine_penalty | -38.157500 | -36.3% | 36.3% | 98.6% |
| proximity | 33.776547 | 32.2% | 32.2% | 100.0% |
| progress | 10.946222 | 10.4% | 21.0% | 100.0% |
| speed_penalty | -10.901868 | -10.4% | 10.4% | 100.0% |
| orientation | -0.173134 | -0.2% | 0.2% | 100.0% |
| contact_encouragement | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
