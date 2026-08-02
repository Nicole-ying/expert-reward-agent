# Training Feedback

## Final-policy outcome
score=-722.255458, len=552.600000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-1851.000051, 4.358282]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity_reward | 357.219844 | 73.0% | 77.1% | 43.9% |
| lateral_drift_penalty | -42.794668 | -8.7% | 8.7% | 52.8% |
| height_health_penalty | -39.911247 | -8.2% | 8.2% | 47.6% |
| action_energy_penalty | -29.244398 | -6.0% | 6.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
