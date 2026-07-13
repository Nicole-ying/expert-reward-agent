# Training Feedback

## Final-policy outcome
score=-24.928910, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-55.543987, 14.434790]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 687.880941 | 51.2% | 51.2% | 100.0% |
| proximity | 645.999266 | 48.1% | 48.1% | 100.0% |
| energy_penalty | -10.000000 | -0.7% | 0.7% | 100.0% |
| terminal_velocity_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
