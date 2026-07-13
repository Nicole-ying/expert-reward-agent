# Training Feedback

## Final-policy outcome
score=-611.764180, len=146.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-1117.411764, -313.257357]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| descent_safety | -7.127694 | -50.1% | 50.1% | 9.9% |
| fuel_penalty | -4.862500 | -34.2% | 34.2% | 66.2% |
| landing_improvement | 2.227192 | 15.7% | 15.7% | 17.2% |
| stable_landed | 0.000000 | 0.0% | 0.0% | 0.0% |
| touchdown_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 13/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
