# Training Feedback

## Final-policy outcome
score=229.814493, len=278.000000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-16.883920, 302.212101]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settled_reward | 222.001056 | 76.0% | 76.0% | 32.6% |
| contact_bonus | 50.203315 | 17.2% | 17.2% | 0.6% |
| goal_proximity | -10.080279 | -3.4% | 3.4% | 100.0% |
| velocity_penalty | -3.773904 | -1.3% | 1.3% | 96.4% |
| landing_approach | 3.580217 | 1.2% | 1.2% | 100.0% |
| fuel_penalty | -1.162250 | -0.4% | 0.4% | 83.6% |
| orientation_penalty | -0.957769 | -0.3% | 0.3% | 100.0% |
| contact_speed_penalty | -0.447208 | -0.2% | 0.2% | 29.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
