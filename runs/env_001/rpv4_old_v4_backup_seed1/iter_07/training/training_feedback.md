# Training Feedback

## Final-policy outcome
score=-5.371040, len=964.100000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-37.696217, 118.125687]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 95.903507 | 86.1% | 86.1% | 0.7% |
| upright_penalty | -5.454539 | -4.9% | 4.9% | 100.0% |
| goal_proximity | 4.753808 | 4.3% | 4.8% | 100.0% |
| velocity_penalty | -4.620901 | -4.1% | 4.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
