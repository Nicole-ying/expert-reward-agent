# Training Feedback

## Final-policy outcome
score=238.545058, len=1189.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-45.196538, 291.157353]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 556.756231 | 90.5% | 90.5% | 100.0% |
| energy_penalty | -35.573389 | -5.8% | 5.8% | 100.0% |
| torso_stability | -22.434003 | -3.6% | 3.6% | 100.0% |
| vertical_velocity_penalty | -0.560140 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
