# Training Feedback

## Final-policy outcome
score=16.493080, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-34.957828, 92.668232]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stable_landing_bonus | 1649.524493 | 51.8% | 51.8% | 96.5% |
| proximity_reward | 1384.676214 | 43.5% | 43.5% | 100.0% |
| distance_penalty | -88.176930 | -2.8% | 2.8% | 100.0% |
| engine_penalty | -29.187000 | -0.9% | 0.9% | 97.3% |
| progress_delta | 20.170868 | 0.6% | 0.7% | 100.0% |
| velocity_damping | -8.972703 | -0.3% | 0.3% | 100.0% |
| orientation_penalty | -0.227809 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
