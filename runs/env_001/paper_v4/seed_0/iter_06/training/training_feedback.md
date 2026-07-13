# Training Feedback

## Final-policy outcome
score=-18.279889, len=911.600000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-45.840393, 57.307131]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| safe_proximity | 662.422072 | 84.1% | 84.1% | 100.0% |
| descent_safety | -92.527172 | -11.8% | 11.8% | 65.7% |
| fuel_penalty | -31.375000 | -4.0% | 4.0% | 68.8% |
| touchdown_bonus | 0.527371 | 0.1% | 0.1% | 0.0% |
| stable_landed | 0.509692 | 0.1% | 0.1% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
