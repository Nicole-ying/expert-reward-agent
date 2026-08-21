# Training Feedback

## Final-policy outcome
score=296.080922, len=1173.600000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[292.278708, 297.862136]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 504.922992 | 95.5% | 95.5% | 100.0% |
| torso_stability | -23.701484 | -4.5% | 4.5% | 100.0% |
| vertical_velocity_penalty | -0.269585 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
