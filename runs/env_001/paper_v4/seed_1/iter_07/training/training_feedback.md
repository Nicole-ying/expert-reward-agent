# Training Feedback

## Final-policy outcome
score=24.338936, len=575.850000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[-492.143659, 247.188147]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 390.287154 | 98.1% | 98.1% | 6.3% |
| shaped_progress | 3.494042 | 0.9% | 1.3% | 99.2% |
| angular_vel_penalty | -2.144577 | -0.5% | 0.5% | 97.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
