# Training Feedback

## Final-policy outcome
score=120.209098, len=819.250000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-34.998420, 234.023251]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 125.190069 | 77.7% | 77.7% | 18.4% |
| descent_incentive | 18.771801 | 11.6% | 11.6% | 80.0% |
| goal_proximity | 7.920113 | 4.9% | 5.1% | 99.5% |
| velocity_penalty | -5.826831 | -3.6% | 3.6% | 99.5% |
| orientation_penalty | -3.183935 | -2.0% | 2.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
