# Training Feedback

## Final-policy outcome
score=260.714949, len=341.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[230.878671, 296.622290]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 262.885847 | 88.7% | 88.7% | 100.0% |
| tilt_penalty | -13.043223 | -4.4% | 4.4% | 100.0% |
| fuel_penalty | -10.005000 | -3.4% | 3.4% | 58.6% |
| velocity_penalty | -8.322307 | -2.8% | 2.8% | 99.7% |
| rotation_penalty | -2.006625 | -0.7% | 0.7% | 99.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
