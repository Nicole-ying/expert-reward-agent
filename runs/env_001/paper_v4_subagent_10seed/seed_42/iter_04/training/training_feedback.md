# Training Feedback

## Final-policy outcome
score=146.416659, len=812.950000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[36.722380, 212.679957]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | 69.212113 | 98.0% | 98.0% | 12.5% |
| safe_progress | 1.382024 | 2.0% | 2.0% | 70.3% |
| orientation_penalty | -0.055110 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
