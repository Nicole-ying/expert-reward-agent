# Training Feedback

## Final-policy outcome
score=141.859009, len=960.500000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-56.757472, 185.622308]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_bonus | 379.242969 | 98.8% | 98.8% | 79.2% |
| velocity_damping | -2.567614 | -0.7% | 0.7% | 100.0% |
| distance_progress | 1.365832 | 0.4% | 0.4% | 100.0% |
| orientation_penalty | -0.636153 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
