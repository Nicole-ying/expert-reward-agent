# Training Feedback

## Final-policy outcome
score=-114.375751, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-138.749698, -99.234408]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_landing_bonus | 30.000000 | 83.4% | 83.4% | 0.4% |
| progress_shaping | 5.596924 | 15.6% | 16.1% | 100.0% |
| action_cost | -0.097000 | -0.3% | 0.3% | 7.1% |
| angle_penalty | -0.086575 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
