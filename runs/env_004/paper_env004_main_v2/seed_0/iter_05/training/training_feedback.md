# Training Feedback

## Final-policy outcome
score=2080.003705, len=889.250000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[722.462765, 2794.029124]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_posture_reward | 1179.447984 | 88.9% | 89.0% | 100.0% |
| vertical_reward | 135.003660 | 10.2% | 10.2% | 45.9% |
| ang_vel_penalty | 11.012731 | 0.8% | 0.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
