# Training Feedback

## Final-policy outcome
score=-64.224908, len=70.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-98.598884, -34.943701]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_pressure | -26.833816 | -82.8% | 82.8% | 100.0% |
| progress | 2.225617 | 6.9% | 7.1% | 100.0% |
| velocity_penalty | -1.785585 | -5.5% | 5.5% | 99.9% |
| contact_state | 1.155000 | 3.6% | 3.6% | 3.4% |
| orientation_penalty | -0.334188 | -1.0% | 1.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 17/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
