# Training Feedback

## Final-policy outcome
score=-46.512046, len=740.150000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[-274.011265, 201.880974]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_landing_bonus | 1195.000000 | 97.7% | 97.7% | 1.6% |
| action_cost | -13.469000 | -1.1% | 1.1% | 91.0% |
| progress_shaping | 5.104587 | 0.4% | 0.7% | 99.9% |
| angle_penalty | -6.827954 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
