# Training Feedback

## Final-policy outcome
score=-15.006998, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-43.536337, 26.845598]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 7.881157 | 40.5% | 43.1% | 100.0% |
| velocity_penalty | -6.826279 | -35.1% | 35.1% | 100.0% |
| orientation_penalty | -3.614935 | -18.6% | 18.6% | 100.0% |
| descent_incentive | 0.624328 | 3.2% | 3.2% | 66.1% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
