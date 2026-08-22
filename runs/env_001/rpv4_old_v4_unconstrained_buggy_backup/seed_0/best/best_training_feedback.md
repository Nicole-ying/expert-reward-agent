# Training Feedback

## Final-policy outcome
score=120.173039, len=467.350000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-109.888883, 244.605909]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 4268.223332 | 98.4% | 98.4% | 10.4% |
| action_penalty | -44.657500 | -1.0% | 1.0% | 93.8% |
| progress_reward | 12.668720 | 0.3% | 0.3% | 99.7% |
| vel_damping | -7.400824 | -0.2% | 0.2% | 99.9% |
| orientation_penalty | -1.094774 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
