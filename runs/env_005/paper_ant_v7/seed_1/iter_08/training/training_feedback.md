# Training Feedback

## Final-policy outcome
score=-311.983426, len=138.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-1027.655662, -117.973598]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_speed_reward | 207.596576 | 95.2% | 95.2% | 100.0% |
| upright_penalty | -10.522678 | -4.8% | 4.8% | 99.7% |
| height_penalty | -0.000485 | -0.0% | 0.0% | 1.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 14/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
