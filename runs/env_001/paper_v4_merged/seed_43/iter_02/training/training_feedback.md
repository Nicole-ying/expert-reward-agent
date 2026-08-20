# Training Feedback

## Final-policy outcome
score=-117.484156, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-137.762360, -103.810338]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_shaping | 1.059252 | 90.2% | 97.2% | 100.0% |
| action_cost | -0.032500 | -2.8% | 2.8% | 4.8% |
| angle_hinge | -0.000440 | -0.0% | 0.0% | 0.1% |
| danger_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
