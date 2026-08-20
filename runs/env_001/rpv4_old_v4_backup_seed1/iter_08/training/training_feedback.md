# Training Feedback

## Final-policy outcome
score=-76.197309, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-117.545539, -12.319018]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 2.728712 | 18.3% | 36.5% | 100.0% |
| velocity_penalty | -5.308256 | -35.5% | 35.5% | 100.0% |
| upright_penalty | -4.189605 | -28.0% | 28.0% | 100.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
