# Training Feedback

## Final-policy outcome
score=175.744060, len=588.650000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-4.793600, 279.181440]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 1177.023470 | 78.0% | 78.0% | 5.8% |
| settled_reward | 297.461290 | 19.7% | 19.7% | 38.3% |
| landing_approach | 14.621929 | 1.0% | 1.0% | 100.0% |
| goal_proximity | -9.891942 | -0.7% | 0.7% | 100.0% |
| velocity_penalty | -5.268320 | -0.3% | 0.3% | 99.3% |
| fuel_penalty | -1.782250 | -0.1% | 0.1% | 60.6% |
| orientation_penalty | -1.407820 | -0.1% | 0.1% | 100.0% |
| contact_speed_penalty | -0.556098 | -0.0% | 0.0% | 38.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
