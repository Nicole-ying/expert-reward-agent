# Training Feedback

## Final-policy outcome
score=27.716998, len=495.550000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-231.405833, 208.200486]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 312.242536 | 90.4% | 90.4% | 12.8% |
| shaped_progress | 20.524081 | 5.9% | 9.5% | 99.8% |
| angular_vel_penalty | -0.364231 | -0.1% | 0.1% | 99.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
