# Training Feedback

## Final-policy outcome
score=-111.178413, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.079360, -95.165512]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_shaping | 5.602874 | 95.1% | 98.5% | 100.0% |
| action_cost | -0.067000 | -1.1% | 1.1% | 4.9% |
| angle_penalty | -0.020036 | -0.3% | 0.3% | 100.0% |
| success_landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
