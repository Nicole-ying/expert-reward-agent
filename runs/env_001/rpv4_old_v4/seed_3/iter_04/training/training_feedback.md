# Training Feedback

## Final-policy outcome
score=183.523455, len=710.250000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[90.971361, 220.088340]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 114.082823 | 86.3% | 86.3% | 4.9% |
| goal_proximity | 7.951091 | 6.0% | 6.2% | 98.1% |
| velocity_penalty | -5.735901 | -4.3% | 4.3% | 98.4% |
| orientation_penalty | -3.386074 | -2.6% | 2.6% | 100.0% |
| descent_incentive | 0.779257 | 0.6% | 0.6% | 72.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
