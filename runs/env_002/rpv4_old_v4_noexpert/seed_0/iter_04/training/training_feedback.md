# Training Feedback

## Final-policy outcome
score=298.557735, len=1043.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[294.558503, 301.707862]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 587.302931 | 95.0% | 95.0% | 100.0% |
| torso_stability | -30.569268 | -4.9% | 4.9% | 100.0% |
| vertical_velocity_penalty | -0.147143 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
