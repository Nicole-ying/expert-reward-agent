# Training Feedback

## Final-policy outcome
score=192.432569, len=1122.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-50.754332, 280.130356]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 333.029013 | 98.0% | 98.5% | 100.0% |
| torso_stability | -4.942750 | -1.5% | 1.5% | 100.0% |
| vertical_velocity_penalty | -0.303253 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
