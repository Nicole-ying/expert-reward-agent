# Training Feedback

## Final-policy outcome
score=227.643558, len=374.600000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-11.906847, 281.988416]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 36.599581 | 55.8% | 55.8% | 20.3% |
| descent_incentive | 11.552093 | 17.6% | 17.6% | 87.9% |
| goal_proximity | 7.917188 | 12.1% | 12.5% | 96.4% |
| velocity_penalty | -6.845214 | -10.4% | 10.4% | 97.0% |
| orientation_penalty | -2.401008 | -3.7% | 3.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
