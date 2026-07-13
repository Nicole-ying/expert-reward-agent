# Training Feedback

## Final-policy outcome
score=-36.731129, len=927.600000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-116.230328, 11.719047]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | -346.944133 | -84.8% | 84.8% | 100.0% |
| velocity_penalty | -37.305640 | -9.1% | 9.1% | 100.0% |
| stability_penalty | -14.771480 | -3.6% | 3.6% | 100.0% |
| contact_reward | 10.200000 | 2.5% | 2.5% | 0.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
